import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import torch
from transformers import Lfm2Config,Lfm2ForCausalLM
from name_decoder.internal_dev import split_internal_rows,evaluate_dev,restore_adapter
from name_decoder.lora import LoRALinear,adapter_state_dict
from name_decoder.noisy_channel import ChannelExample

ROOT=Path(__file__).resolve().parents[1]

class InternalDevTests(unittest.TestCase):
    def test_vessel_split_reproducible_all_descendants(self):
        rows=pd.DataFrame([dict(sample_id=f'{v}_{i}',target_vessel_id=str(v),hangul=v%2==0)
                           for v in range(40) for i in range(3)])
        a,b=split_internal_rows(rows,.1,0,42)
        c,d=split_internal_rows(rows.sample(frac=1,random_state=7),.1,0,42)
        self.assertEqual(set(b.target_vessel_id),set(d.target_vessel_id))
        self.assertFalse(set(a.target_vessel_id)&set(b.target_vessel_id))
        self.assertEqual(len(a)+len(b),len(rows))
        with self.assertRaises(ValueError):split_internal_rows(rows,.1,1)

    def test_cpu_snapshot_is_immutable_and_restore(self):
        model=torch.nn.Sequential(LoRALinear(torch.nn.Linear(2,2),rank=1))
        state=adapter_state_dict(model)
        initial=state['0.lora_A'].clone()
        with torch.no_grad():model[0].lora_A.add_(2)
        torch.testing.assert_close(state['0.lora_A'],initial)
        restore_adapter(model,state)
        torch.testing.assert_close(model[0].lora_A,initial)
        with self.assertRaises(ValueError):restore_adapter(model,{})

    def test_dev_row_token_metrics_and_mode_restoration(self):
        class Fake(torch.nn.Module):
            def forward(self,input_ids,**kwargs):
                logits=torch.zeros(*input_ids.shape,8)
                logits[:,:,2]=2
                return SimpleNamespace(logits=logits)
        model=Fake().train()
        groups=[[ChannelExample((1,2,2),1),ChannelExample((3,2,2),1)],
                [ChannelExample((1,4,4,4,4),1),ChannelExample((3,4,4,4,4),1)]]
        original=groups
        groups=[original[0],original[1],original[1]]
        flags=[True,True,False]
        row=evaluate_dev(model,groups,flags,0,2,'cpu',.25,1.,'row',.7)
        token=evaluate_dev(model,groups,flags,0,1,'cpu',.25,1.,'token',.7)
        self.assertTrue(model.training)
        self.assertTrue(torch.isfinite(torch.tensor([row,token])).all())
        self.assertNotAlmostEqual(row,token,places=5)
        # Repeat within one language: dataset metric must not depend on batching.
        for reduction in ['row','token']:
            x=evaluate_dev(model,groups*2,flags*2,0,1,'cpu',.25,1.,reduction,.7)
            y=evaluate_dev(model,groups*2,flags*2,0,3,'cpu',.25,1.,reduction,.7)
            self.assertAlmostEqual(x,y,places=6)
        class Bad(Fake):
            def forward(self,input_ids,**kwargs):
                return SimpleNamespace(logits=torch.full((*input_ids.shape,8),float('nan')))
        bad=Bad().train()
        with self.assertRaises(FloatingPointError):evaluate_dev(bad,groups,flags,0,2,'cpu',.1,1.,'row',.7)
        self.assertTrue(bad.training)

    def test_miner_uses_only_internal_train_and_crossfit(self):
        from name_decoder.internal_dev import crossfit_confusions
        from name_decoder.catalog import CatalogNames
        rows=pd.DataFrame([dict(sample_id=f's{i}',target_vessel_id=str(i),hangul=i%2==0,fold=i%4+1,vessel_name=f'N{i}') for i in range(40)])
        train,dev=split_internal_rows(rows,.1,4,42)
        seen=[]
        def fit(learning,*args,**kwargs):
            seen.append(set(learning.sample_id))
            return None
        def predict(probe,*args,**kwargs):
            self.assertNotIn('target_vessel_id',probe.columns)
            self.assertNotIn('fold',probe.columns)
            return [],[],[[(1.,'0'),(.5,'1')] for _ in range(len(probe))]
        engine=SimpleNamespace(CatalogIndex=lambda x:None,fit_matching_model=fit,predict_rows=predict)
        loader=SimpleNamespace(exec_module=lambda m:None)
        spec=SimpleNamespace(loader=loader)
        cat=CatalogNames([dict(vessel_id=str(i),vessel_name=f'N{i}') for i in range(40)])
        with patch('importlib.util.spec_from_file_location',return_value=spec),patch('importlib.util.module_from_spec',return_value=engine):
            negatives,audit=crossfit_confusions(train,None,cat)
        self.assertTrue(len(negatives)>0)
        for item,ids in zip(audit,seen):
            self.assertFalse(ids & set(dev.sample_id))
            self.assertFalse(ids & set(item['probe_sample_ids']))
            self.assertEqual(ids,set(item['fit_sample_ids']))

    def test_full_training_saves_restored_best_and_no_dev_draws(self):
        from experiments.train_noisy_channel import parse_args,train
        from experiments.noisy_channel_common import write_json,sha256
        from name_decoder.catalog import CatalogNames
        # A tiny integration fixture, not a performance experiment.
        class Tokenizer:
            pad_token_id=0;eos_token_id=7;all_special_ids=[0,7]
            def encode(self,text,**kwargs):return [3+ord(c)%20 for c in text]
            def apply_chat_template(self,*args,**kwargs):return [1,2]
            def save_pretrained(self,path):
                Path(path).mkdir(exist_ok=True,parents=True)
                (Path(path)/'tokenizer_config.json').write_text('{}')
        rows=pd.DataFrame([dict(sample_id=f's{v}',target_vessel_id=str(v),vessel_name=('SHIP ' if v%2 else '\ubc30 ')+str(v),hangul=v%2==0,fold=1+v%4) for v in range(24)])
        outer=pd.DataFrame([dict(sample_id='outer',target_vessel_id='outside')])
        cat=CatalogNames([dict(vessel_id=str(v),vessel_name='SHIP '+str(v)) for v in range(24)])
        cfg=Lfm2Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,layer_types=['conv','full_attention'],block_multiple_of=8,block_auto_adjust_ff_dim=False)
        cfg._attn_implementation='eager'
        torch.set_num_threads(2)
        for sampling,selection in [('cycle','internal-best'),('replacement','internal-best'),('cycle','last')]:
            with self.subTest(sampling=sampling),tempfile.TemporaryDirectory(dir=ROOT/'scratch') as tmp:
                folder=Path(tmp);base=folder/'base';base.mkdir();(base/'config.json').write_text('{}')
                out=folder/'output';out.mkdir();record_path=folder/'record.json'
                template=json.loads((ROOT/'project_log/templates/experiment.json').read_text(encoding='utf-8-sig'))
                record={**template,'environment':{},'cost':{}}
                with patch('sys.argv',['train','--experiment-id','tiny','--pretrained',str(base),'--device','cpu','--steps','3','--warmup-steps','0','--batch-size','2','--accumulation','1','--rank','2','--alpha','4','--max-length','64','--dev-every','2','--dev-max-vessels','2','--augment-factor','0','--no-learned-hangul','--sampling',sampling,'--checkpoint-selection',selection]):args=parse_args()
                tiny=Lfm2ForCausalLM(cfg)
                # Force step1 as best; step2 must restore step1 adapter bytes.
                with patch('name_decoder.data.read_frames',return_value=(None,outer,rows,cat)),patch('experiments.train_noisy_channel.source_hashes',return_value={'folds':'fake'}),patch('name_decoder.noisy_channel.load_channel_tokenizer',return_value=Tokenizer()),patch('transformers.AutoModelForCausalLM.from_pretrained',return_value=tiny),patch('torch.set_num_interop_threads'),patch('name_decoder.internal_dev.evaluate_dev',side_effect=[1.,2.]):
                    train(args,out,record_path,record)
                saved=json.loads(record_path.read_text())
                self.assertEqual(saved['checkpoint_selection']['best_step'],2 if selection=='internal-best' else None)
                self.assertEqual(saved['checkpoint_selection']['saved_step'],2 if selection=='internal-best' else 3)
                if selection=='internal-best':
                    self.assertEqual(sha256(out/'adapter.safetensors'),sha256(out/'best_adapter.safetensors'))
                history=json.loads((out/'training_history.json').read_text())
                self.assertIsNone(history[0]['best_dev_loss'])
                self.assertIsNone(history[0]['best_step'])
                split=json.loads((out/'internal_split.json').read_text())
                self.assertFalse(set(split['training_vessel_ids'])&set(split['dev_vessel_ids']))
                self.assertEqual(saved['training_pairs']['raw_pairs_count'],len(split['training_sample_ids']))
                manifest=json.loads((out/'model/noisy_channel_manifest.json').read_text())
                self.assertEqual(manifest['checkpoint_selection']['best_step'],2 if selection=='internal-best' else None)
