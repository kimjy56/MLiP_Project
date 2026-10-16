"""Reproducible EDA of the two official training resources; never reads test.csv.

Run: python analysis/eda.py --data-dir data --output-dir analysis/output
Charts use English labels to work without additional Korean fonts.
"""
from pathlib import Path
import argparse
import io
import json
import os
import re
import unicodedata

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vessel-eda-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def normalized(value):
    if not isinstance(value, str):
        return ''
    return ''.join(c for c in unicodedata.normalize('NFKC', value).upper() if c.isalnum())


def audit(train, catalog, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    info = io.StringIO()
    for name, frame in [('train', train), ('catalog', catalog)]:
        info.write(f'\n{name}\n')
        frame.info(buf=info)
        frame.head().to_csv(out / f'{name}_head.csv', index=False)
        frame.describe(include='all').T.to_csv(out / f'{name}_describe.csv')
        pd.DataFrame({'dtype': frame.dtypes.astype(str), 'unique': frame.nunique(),
                      'missing_count': frame.isna().sum(),
                      'missing_rate': frame.isna().mean()}).to_csv(out / f'{name}_schema.csv')
    (out/'df_info.txt').write_text(info.getvalue(), encoding='utf-8')
    assert train.sample_id.notna().all() and train.sample_id.is_unique
    assert catalog.vessel_id.notna().all() and catalog.vessel_id.is_unique
    unknown = (~train.target_vessel_id.isin(catalog.vessel_id)).sum()
    assert unknown == 0
    h = train.vessel_name.str.contains('[가-힣]', na=False)
    joined = train.merge(catalog, left_on='target_vessel_id', right_on='vessel_id',
                         suffixes=('_q', '_c'), validate='many_to_one')
    target_counts = train.target_vessel_id.value_counts()
    target_counts.rename('rows').to_csv(out/'target_counts.csv')
    misscols = ['gross_tonnage', 'international_gross_tonnage', 'flag_name']
    patterns = train[misscols].isna().value_counts().rename('rows').reset_index()
    patterns.to_csv(out/'missing_patterns.csv', index=False)
    lang = np.where(h, 'Hangul', 'Non-Hangul')
    missing_lang = train[misscols].isna().assign(language=lang).groupby('language').mean()
    missing_lang.to_csv(out/'missing_by_language.csv')
    missing_year = train[misscols].isna().assign(year=train.arrival_year).groupby('year').mean()
    missing_year.to_csv(out/'missing_by_year.csv')
    train[misscols].isna().assign(vessel_type=train.vessel_type).groupby('vessel_type').mean().to_csv(out/'missing_by_type.csv')
    chi = {}
    for col in ['gross_tonnage', 'flag_name']:
        chi2, p, _, _ = chi2_contingency(pd.crosstab(h, train[col].isna()))
        chi[col] = {'chi2': chi2, 'p_value': p, 'phi': float(np.sqrt(chi2/len(train))),
                    'warning': 'Exploratory row-level test; repeated vessels violate independence. Not proof of MAR/MNAR.'}
    distributions = []
    for name, frame, cols in [('train', train, ['gross_tonnage', 'international_gross_tonnage']),
                              ('catalog', catalog, ['gross_tonnage', 'deadweight_tonnage'])]:
        for col in cols:
            x = pd.to_numeric(frame[col], errors='coerce').dropna()
            q1, q3 = x.quantile([.25, .75])
            z = np.log1p(x[x>0])
            median = z.median()
            mad = np.median(np.abs(z-median))
            distributions.append({'dataset': name, 'column': col, 'count': len(x),
                'min': x.min(), 'median': x.median(), 'p95': x.quantile(.95),
                'p99': x.quantile(.99), 'max': x.max(), 'skew': x.skew(),
                'excess_kurtosis': x.kurt(), 'log1p_skew': z.skew(),
                'nonpositive': int((x<=0).sum()),
                'iqr_upper': q3+1.5*(q3-q1),
                'iqr_flag_count': int(((x<q1-1.5*(q3-q1))|(x>q3+1.5*(q3-q1))).sum()),
                'log_mad_flag_count': int((np.abs(.6745*(z-median)/mad)>3.5).sum()) if mad>0 else None})
    pd.DataFrame(distributions).to_csv(out/'numeric_distributions.csv', index=False)
    # Mixed-unit size is summarized within units, never pooled for a physical threshold.
    catalog.groupby('unit').size().rename('rows').to_csv(out/'catalog_units.csv')
    catalog.groupby('unit')['size'].agg(['count','min','median','max','skew']).to_csv(out/'size_by_unit.csv')
    numeric = train[['gross_tonnage','international_gross_tonnage','arrival_year']]
    numeric.corr().to_csv(out/'pearson.csv')
    numeric.corr(method='spearman').to_csv(out/'spearman.csv')
    r = numeric.gross_tonnage.corr(numeric.international_gross_tonnage)
    complete = train.gross_tonnage.notna() & train.international_gross_tonnage.notna()
    qg, qi, cg = joined.gross_tonnage_q, joined.international_gross_tonnage, joined.gross_tonnage_c
    gap = pd.concat([abs(qg-cg)/np.maximum(qg,cg), abs(qi-cg)/np.maximum(qi,cg)],axis=1).min(axis=1)
    gap_stats = gap.groupby(lang).describe(percentiles=[.5,.9,.95,.99])
    gap_stats.to_csv(out/'true_pair_tonnage_gap.csv')
    for field in ['vessel_type','flag_name']:
        # Descriptive only: do not reuse these full-data tables as validation features.
        pd.crosstab(joined[field+'_q'].fillna('__MISSING__'),
                    joined[field+'_c'].fillna('__MISSING__'), normalize='index').to_csv(out/f'{field}_crosswalk_EDA_ONLY.csv')
    def group_count(cols):
        return int((train.groupby(cols, dropna=False).target_vessel_id.nunique()>1).sum())
    full_input = [c for c in train.columns if c not in ['sample_id','target_vessel_id']]
    summary = {
        'train_rows': len(train), 'catalog_rows': len(catalog),
        'train_unique_vessels': len(target_counts), 'training_catalog_coverage': len(target_counts)/len(catalog),
        'hangul_rows': int(h.sum()), 'hangul_fraction': float(h.mean()),
        'hangul_vessels': int(train.loc[h,'target_vessel_id'].nunique()),
        'catalog_hangul_rows': int(catalog.vessel_name.str.contains('[가-힣]',na=False).sum()),
        'target_frequency_histogram': {str(k):int(v) for k,v in target_counts.value_counts().sort_index().items()},
        'duplicate_full_inputs': int(train.duplicated(full_input).sum()),
        'conflicting_full_input_groups': group_count(full_input),
        'duplicate_name_tonnage': int(train.duplicated(['vessel_name','gross_tonnage','international_gross_tonnage']).sum()),
        'call_sign_ambiguous': group_count(['call_sign']),
        'normalized_name_ambiguous': int((train.assign(normalized=train.vessel_name.map(normalized)).groupby('normalized').target_vessel_id.nunique()>1).sum()),
        'tonnage_pearson': r, 'tonnage_two_predictor_vif': 1/(1-r*r),
        'both_tonnages_present': int(complete.sum()),
        'equal_tonnages': int(train.loc[complete,'gross_tonnage'].eq(train.loc[complete,'international_gross_tonnage']).sum()),
        'international_only_catalog_exact': int((qi.eq(cg)&~qg.eq(cg)).sum()),
        'gross_only_catalog_exact': int((qg.eq(cg)&~qi.eq(cg)).sum()),
        'true_catalog_gt_missing': int(cg.isna().sum()),
        'normalized_name_exact_rate': float(joined.vessel_name_q.map(normalized).eq(joined.vessel_name_c.map(normalized)).mean()),
        'matched_tonnage_gap_max': float(gap.max()), 'matched_tonnage_gap_over_10pct': int((gap>.1).sum()),
        'tonnage_exact_either': int((qg.eq(cg)|qi.eq(cg)).sum()),
        'catalog_gt_dwt_pearson': catalog.gross_tonnage.corr(catalog.deadweight_tonnage),
        'chi2_missingness_exploratory': chi,
        'temporal_example': {},
    }
    for cut in [2023,2024]:
        past = train[train.arrival_year<=cut]
        future = train[train.arrival_year>cut]
        summary['temporal_example'][str(cut)] = {'train_rows':len(past), 'validation_rows':len(future),
            'seen_vessel_fraction':float(future.target_vessel_id.isin(past.target_vessel_id).mean()),
            'hangul_rows':int(future.vessel_name.str.contains('[가-힣]').sum())}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    blue, orange = '#2563eb', '#f59e0b'
    fig, axes = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True)
    ax = axes[0,0]
    missing_lang.T.plot.bar(ax=ax,color=[orange,blue],rot=0)
    ax.set_xticklabels(['Gross tonnage','International GT','Flag'])
    ax.set_ylabel('Missing fraction'); ax.set_title('Missingness differs by language')
    ax = axes[0,1]
    ax.bar(['Train share','Score weight'],[h.mean(),.7],color=[blue,orange])
    ax.set_ylim(0,1);ax.set_title('Hangul: 9.7% of rows, 70% of score')
    ax = axes[1,0]
    freq=target_counts.value_counts().sort_index()
    ax.bar(freq.index,freq.values,color=blue);ax.set_xticks([1,2,3,4]);ax.set_xlabel('Records per vessel');ax.set_ylabel('Number of vessels');ax.set_title('Many rare targets; repeated records')
    ax = axes[1,1]
    for mask,label,color in [(h,'Hangul',orange),(~h,'Non-Hangul',blue)]:
        ax.hist(gap[mask].dropna()*100,bins=np.linspace(0,10,41),histtype='step',density=True,label=label,color=color,lw=2)
    ax.set_xlabel('Best relative GT difference to TRUE vessel (%)');ax.set_ylabel('Density');ax.set_title('Tonnage agreement: useful, not sufficient');ax.legend()
    fig.suptitle('Vessel matching | Data quality and evaluation priorities',fontsize=16)
    fig.savefig(out/'eda_overview.png',dpi=170);plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,(label,frame) in enumerate([('Train',train),('Catalog',catalog)]):
        x=frame.gross_tonnage.dropna()
        axes[row,0].hist(x,bins=60,color=blue);axes[row,0].set_title(f'{label}: raw gross tonnage');axes[row,0].set_ylabel('Rows')
        axes[row,1].hist(np.log1p(x),bins=60,color=orange);axes[row,1].set_title(f'{label}: log1p(gross tonnage)')
    fig.savefig(out/'tonnage_distributions.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    corr=numeric.corr();im=axes[0].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    labs=['Gross GT','International GT','Arrival year']
    axes[0].set_xticks(range(3),labs,rotation=25);axes[0].set_yticks(range(3),labs)
    for i in range(3):
        for j in range(3):axes[0].text(j,i,f'{corr.iloc[i,j]:.5f}',ha='center',va='center')
    axes[0].set_title('Pearson correlation (pairwise complete)');fig.colorbar(im,ax=axes[0],shrink=.7)
    missing_year.plot(ax=axes[1],marker='o');axes[1].set_title('Missingness by arrival year');axes[1].set_ylabel('Missing fraction')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'correlation_and_year.png',dpi=170);plt.close(fig)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output')
    args=parser.parse_args()
    result=audit(pd.read_csv(args.data_dir/'train.csv'),pd.read_csv(args.data_dir/'vessel_catalog.csv'),args.output_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))
