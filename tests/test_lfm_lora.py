import tempfile
import unittest
from pathlib import Path
import torch
from transformers import Lfm2Config,Lfm2ForCausalLM
from name_decoder.lora import inject_attention_lora,merge_attention_lora

class LfmCompatibility(unittest.TestCase):
    def test_local_lfm_tokenizer_contract(self):
        from name_decoder.noisy_channel import load_channel_tokenizer,encode_channel_example
        checkpoint=Path(__file__).resolve().parents[1]/'artifacts/pretrained/lfm2.5-350m'
        if not (checkpoint/'tokenizer.json').exists():
            self.skipTest('LFM tokenizer not downloaded')
        tokenizer=load_channel_tokenizer(checkpoint)
        observed= '\uc528 \uc2a4\ud0c0'
        example=encode_channel_example(tokenizer,'SEA STAR',observed)
        self.assertEqual(tokenizer.eos_token_id,7)
        self.assertEqual(tokenizer.decode(example.input_ids[example.prompt_length:-1]),observed)
        self.assertEqual(example.input_ids[-1],7)

    def test_hybrid_training_merge_reload(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config=Lfm2Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
            layer_types=['conv','full_attention','conv','full_attention'],
            block_multiple_of=16,block_auto_adjust_ff_dim=False)
        config._attn_implementation='eager'
        model=Lfm2ForCausalLM(config)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device);model.config.use_cache=False
        spec=inject_attention_lora(model,rank=4,alpha=8)
        self.assertEqual(spec['attention_layer_count'],2)
        self.assertEqual(len(spec['modules']),4)
        inputs=torch.randint(3,64,(2,12),device=device)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.001)
        loss=model(input_ids=inputs,labels=inputs,use_cache=False).loss
        loss.backward();optimizer.step();optimizer.zero_grad()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(model.model.layers[0].conv.in_proj.weight.requires_grad)
        model.eval()
        with torch.no_grad():before=model(input_ids=inputs,use_cache=False).logits
        self.assertEqual(merge_attention_lora(model),4)
        with torch.no_grad():after=model(input_ids=inputs,use_cache=False).logits
        torch.testing.assert_close(before,after,atol=1e-5,rtol=1e-4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'scratch') as d:
            model.to('cpu').save_pretrained(d)
            loaded=Lfm2ForCausalLM.from_pretrained(d,attn_implementation='eager').to(device).eval()
            with torch.no_grad():reloaded=loaded(input_ids=inputs,use_cache=False).logits
            torch.testing.assert_close(after,reloaded,atol=1e-5,rtol=1e-4)

if __name__=='__main__':unittest.main()
