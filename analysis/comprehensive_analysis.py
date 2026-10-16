"""Additional integrity, representation and saved-validation diagnostics.

Reads official training/catalog data only. Never reads test.csv or trains a model.
Run from project root: python analysis/comprehensive_analysis.py
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import unicodedata

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vessel-eda-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize(x):
    return ''.join(c for c in unicodedata.normalize('NFKC',x).upper() if c.isalnum())


def run(data_dir, reports_dir, output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(data_dir/'train.csv')
    catalog=pd.read_csv(data_dir/'vessel_catalog.csv')
    labeled=catalog[catalog.vessel_id.isin(train.target_vessel_id)].copy()
    checks={}
    numeric={'train':['gross_tonnage','international_gross_tonnage','arrival_year'],
             'catalog':['size','deadweight_tonnage','gross_tonnage']}
    for name,df,filename in [('train',train,'train.csv'),('catalog',catalog,'vessel_catalog.csv')]:
        raw=pd.read_csv(data_dir/filename,dtype=str,keep_default_na=False)
        audit=[]
        for col in df:
            source=raw[col].str.strip()
            number=pd.to_numeric(source.where(source.ne('')),errors='coerce') if col in numeric[name] else None
            audit.append({'column':col,'dtype':str(df[col].dtype),'unique':int(df[col].nunique()),
                'missing_count':int(df[col].isna().sum()),'missing_rate':float(df[col].isna().mean()),
                'constant_nonmissing':bool(df[col].nunique(dropna=True)<=1),
                'blank_or_whitespace':int(source.eq('').sum()),
                'trim_changes':int(raw[col].ne(source).sum()),
                'numeric_parse_failures':int((source.ne('')&number.isna()).sum()) if number is not None else None,
                'nonfinite_numeric':int((number.notna()&~np.isfinite(number)).sum()) if number is not None else None,
                'nonpositive_numeric':int(number.le(0).sum()) if number is not None else None})
        pd.DataFrame(audit).to_csv(output_dir/f'{name}_integrity.csv',index=False)
        checks[name]={'rows':len(df),'columns':len(df.columns),'exact_duplicate_rows':int(df.duplicated().sum()),
                      'constant_columns':[col for col in df if df[col].nunique(dropna=True)<=1]}
    checks['year_noninteger']=int((train.arrival_year%1!=0).sum())
    checks['year_range']=[int(train.arrival_year.min()),int(train.arrival_year.max())]
    norm_counts=catalog.vessel_name.map(normalize).value_counts()
    checks['catalog_normalized_name_ambiguous_groups']=int(norm_counts.gt(1).sum())
    checks['catalog_normalized_name_duplicate_excess_rows']=int(catalog.vessel_name.map(normalize).duplicated().sum())
    checks['catalog_normalized_name_max_multiplicity']=int(norm_counts.max())
    # Compare size with DWT only when unit is DWT and BOTH values are known.
    mask=catalog.unit.eq('DWT')&catalog['size'].notna()&catalog.deadweight_tonnage.notna()
    checks['dwt_size_comparison']={'complete_rows':int(mask.sum()),
        'unequal_rows':int(catalog.loc[mask,'size'].ne(catalog.loc[mask,'deadweight_tonnage']).sum())}
    profiles={}
    for col in ['vessel_type','flag_name']:
        profile=pd.DataFrame({'catalog_ships':catalog[col].value_counts(),
                              'labeled_ships':labeled[col].value_counts()}).fillna(0).astype(int)
        profile['catalog_share']=profile.catalog_ships/len(catalog)
        profile['labeled_share']=profile.labeled_ships/len(labeled)
        profile['labeled_coverage']=profile.labeled_ships/profile.catalog_ships
        profile.sort_values('labeled_ships',ascending=False).to_csv(output_dir/f'{col}_representation.csv')
        profiles[col]=profile
    yearly=train.assign(hangul=train.vessel_name.str.contains('[가-힣]',na=False),
        gt_missing=train.gross_tonnage.isna(),flag_missing=train.flag_name.isna()).groupby('arrival_year').agg(
        rows=('sample_id','size'),ships=('target_vessel_id','nunique'),hangul_rows=('hangul','sum'),
        hangul_fraction=('hangul','mean'),gt_missing_fraction=('gt_missing','mean'),
        flag_missing_fraction=('flag_missing','mean'))
    yearly.to_csv(output_dir/'year_representation.csv')
    for col in ['flag_name','vessel_type']:
        train.groupby(col,dropna=False).agg(rows=('sample_id','size'),ships=('target_vessel_id','nunique')).sort_values('rows',ascending=False).to_csv(output_dir/f'query_{col}_groups.csv')
    representation={
        'catalog_gt_median':float(catalog.gross_tonnage.median()),
        'labeled_unique_ship_gt_median':float(labeled.gross_tonnage.median()),
        'train_record_gt_median':float(train.gross_tonnage.median()),
        'labeled_ship_count':len(labeled),'catalog_ship_count':len(catalog),
        'single_record_vessel_fraction':float(train.target_vessel_id.value_counts().eq(1).mean()),
    }
    for col,p in profiles.items():
        representation[col]={'catalog_categories':len(p),'labeled_categories':int(p.labeled_ships.gt(0).sum()),
            'unlabeled_categories':int(p.labeled_ships.eq(0).sum()),
            'labeled_categories_with_1_to_5_ships':int(p.labeled_ships.between(1,5).sum())}

    # Diagnostic expansion of predictions saved in the previous model experiment.
    # It does not claim a new training run or create a fresh untouched test set.
    metrics={}
    for split in ['vessel_holdout','row_holdout']:
        pred_path=reports_dir/f'{split}_predictions.csv'
        if not pred_path.exists():
            metrics[split]={'available':False}
            continue
        pred=pd.read_csv(pred_path)
        joined=pred.merge(train,on='sample_id',validate='one_to_one')
        assert joined.truth.eq(joined.target_vessel_id).all()
        joined['correct']=joined.truth.eq(joined.prediction)
        joined['language']=np.where(joined.vessel_name.str.contains('[가-힣]',na=False),'Hangul','Non-Hangul')
        joined['tonnage_missing']=joined.gross_tonnage.isna()&joined.international_gross_tonnage.isna()
        for group in ['language','tonnage_missing','vessel_type','arrival_year']:
            stats=joined.groupby(group,dropna=False).agg(rows=('sample_id','size'),vessels=('truth','nunique'),
                accuracy=('correct','mean'),errors=('correct',lambda x:int((~x).sum())))
            stats.to_csv(output_dir/f'{split}_by_{group}.csv')
        by_vessel=joined.groupby('truth').correct.mean()
        by_language=joined.groupby('language').correct.mean()
        metrics[split]={'available':True,'source':'saved predictions of prior LightGBM experiment',
            'rows':len(joined),'vessels':joined.truth.nunique(),'error_rows':int((~joined.correct).sum()),
            'vessels_with_any_error':joined.loc[~joined.correct,'truth'].nunique(),
            'record_accuracy':float(joined.correct.mean()),'macro_vessel_accuracy':float(by_vessel.mean()),
            'hangul_accuracy':float(by_language['Hangul']),
            'non_hangul_accuracy':float(by_language['Non-Hangul']),
            'weighted_accuracy':float(.7*by_language['Hangul']+.3*by_language['Non-Hangul'])}

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True)
    blue,orange='#2563eb','#f59e0b'
    bins=np.linspace(4.5,13,55)
    axes[0,0].hist(np.log1p(catalog.gross_tonnage.dropna()),bins=bins,density=True,histtype='step',label='All catalog ships',color=blue,lw=2)
    axes[0,0].hist(np.log1p(labeled.gross_tonnage.dropna()),bins=bins,density=True,histtype='step',label='Unique labeled ships',color=orange,lw=2)
    axes[0,0].set_title('Labeled ships are larger than the catalog median');axes[0,0].set_xlabel('log1p(catalog gross tonnage)');axes[0,0].set_ylabel('Density');axes[0,0].legend(fontsize=9)
    types=profiles['vessel_type'].sort_values('labeled_ships',ascending=False).head(8)
    types[['catalog_share','labeled_share']].rename(columns={'catalog_share':'All catalog','labeled_share':'Labeled ships'}).plot.barh(ax=axes[0,1],color=[blue,orange])
    axes[0,1].invert_yaxis();axes[0,1].set_ylabel('');axes[0,1].set_xlabel('Share of ships');axes[0,1].set_title('Type composition differs (top 8 labeled types)')
    axes[1,0].bar(yearly.index,yearly.rows,color=blue);axes[1,0].set_title('Records by arrival year');axes[1,0].set_xlabel('Year');axes[1,0].set_ylabel('Records');axes[1,0].ticklabel_format(axis='x',style='plain',useOffset=False)
    axes[1,1].plot(yearly.index,yearly.hangul_fraction*100,'o-',color=orange,label='Hangul records')
    axes[1,1].plot(yearly.index,yearly.gt_missing_fraction*100,'o-',color=blue,label='Missing tonnage')
    axes[1,1].set_title('Language mix and missingness vary over years');axes[1,1].set_xlabel('Year');axes[1,1].set_ylabel('Percent of records');axes[1,1].legend(fontsize=9)
    fig.suptitle('Representation audit | Catalog comparison is not a test-distribution estimate',fontsize=14)
    fig.savefig(output_dir/'representation.png',dpi=160);plt.close(fig)
    result={'integrity':checks,'representation':representation,'prior_model_diagnostics':metrics,
        'input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [data_dir/'train.csv',data_dir/'vessel_catalog.csv']},
        'limitations':['No hidden test data','No port/route/location columns','Unknown sampling and observation cutoff',
                       'No new model trained; validation metrics reuse earlier saved predictions']}
    (output_dir/'comprehensive_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=root/'data')
    parser.add_argument('--reports-dir',type=Path,default=root/'reports')
    parser.add_argument('--output-dir',type=Path,default=root/'analysis/output')
    args=parser.parse_args()
    run(args.data_dir,args.reports_dir,args.output_dir)
