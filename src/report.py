"""Generate publication figures, an HTML results explorer, notebook and full README."""
import html
import json
from pathlib import Path
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .features import FEATURES, PROFILE_FEATURES

FIG = Path('reports/figures')
COLORS = ['#183b56','#d77b35','#27857f','#766895','#a25761','#7f8c65']

def table(frame):
    frame=frame.copy().astype(str)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in frame.values)

def save(name, caption='Source: Transfermarkt via dcaribou · May–July 2025 target cohort'):
    fig=plt.gcf()
    fig.text(.01,.012,caption,fontsize=8,color='#56616d')
    fig.tight_layout(rect=[0,.045,1,1])
    fig.savefig(FIG/f'{name}.png',dpi=170,facecolor='white')
    fig.savefig(FIG/f'{name}.svg',facecolor='white')
    plt.close(fig)

def transfer_analysis(test):
    path=Path('data/raw/transfers.csv.gz')
    transfers=pd.read_csv(path,parse_dates=['transfer_date'])
    transfers=transfers.drop(columns=['market_value_in_eur'],errors='ignore')
    transfers['transfer_fee']=pd.to_numeric(transfers.transfer_fee,errors='coerce')
    tx=transfers.loc[(transfers.transfer_date>=pd.Timestamp('2025-07-01')) &
        (transfers.transfer_date<pd.Timestamp('2025-10-01')) & (transfers.transfer_fee>0)].copy()
    # Latest known value strictly before the transaction, rather than summer target.
    vals=pd.read_csv('data/raw/player_valuations.csv.gz',parse_dates=['date'])
    vals=vals.loc[vals.market_value_in_eur>0,['player_id','date','market_value_in_eur']]
    tx=pd.merge_asof(tx.sort_values('transfer_date'),vals.sort_values('date'),
        left_on='transfer_date',right_on='date',by='player_id',direction='backward',allow_exact_matches=False)
    features=pd.read_csv('data/processed/player_seasons.csv')
    features=features.loc[features.year==2025,['player_id','age','position','goals_per90','assists_per90','minutes_played','last_match_date']]
    tx=tx.merge(features,on='player_id',validate='many_to_one').dropna(subset=['market_value_in_eur'])
    tx=tx.drop_duplicates(['player_id','transfer_date','from_club_id','to_club_id'])
    columns=['transfer_fee','market_value_in_eur','age','minutes_played','goals_per90','assists_per90']
    corr=tx[columns].corr(method='spearman')['transfer_fee'].drop('transfer_fee').sort_values(ascending=False)
    tx.to_csv('reports/transfer_analysis.csv',index=False)
    result={'paid_transfers':len(tx),'spearman_with_fee':corr.to_dict(),
            'median_fee_eur':float(tx.transfer_fee.median()),
            'median_fee_to_prior_value_ratio':float((tx.transfer_fee/tx.market_value_in_eur).median())}
    Path('reports/transfer_results.json').write_text(json.dumps(result,indent=2))
    plt.figure(figsize=(8,5))
    plt.scatter(tx.market_value_in_eur/1e6,tx.transfer_fee/1e6,s=22,alpha=.65,color=COLORS[0])
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Last pre-transfer market estimate (€ million, log scale)')
    plt.ylabel('Reported paid transfer fee (€ million, log scale)')
    plt.title(f'Actual fees versus pre-transfer estimates · {len(tx)} transactions')
    save('transfer_fees','Source: Transfermarkt via dcaribou · paid transfers, July–September 2025; five-league feature cohort')
    return result

def report():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
        'axes.titlesize':14,'axes.labelsize':10,'figure.figsize':(9,5),'axes.prop_cycle':plt.cycler(color=COLORS)})
    m=json.loads(Path('reports/metrics.json').read_text())
    audit=json.loads(Path('reports/data_audit.json').read_text())
    data=pd.read_csv('data/processed/player_seasons.csv')
    test=pd.read_csv('reports/player_profiles.csv')
    imp=pd.read_csv('reports/importance.csv')
    clusters=pd.read_csv('reports/cluster_summary.csv')
    tx=transfer_analysis(test)
    plt.figure(); plt.hist(test.market_value_in_eur/1e6,bins=np.geomspace(.01,250,40),color=COLORS[0])
    plt.xscale('log');plt.xlabel('Market estimate (€ million, log scale)');plt.ylabel('Player count');plt.title('Market values have a long upper tail');save('value_distribution')
    plt.figure();plt.scatter(test.age,test.market_value_in_eur/1e6,alpha=.3,s=14)
    plt.yscale('log');plt.xlabel('Age at 1 May 2025 (years)');plt.ylabel('Market estimate (€ million, log scale)');plt.title('Age and market estimate · descriptive association');save('age_value')
    groups=list(test.groupby('position'))
    plt.figure();plt.boxplot([g.market_value_in_eur/1e6 for _,g in groups],tick_labels=[n for n,_ in groups],showfliers=False)
    plt.yscale('log');plt.xlabel('Position from profile snapshot');plt.ylabel('Market estimate (€ million, log scale)');plt.title('Market values by position · outliers hidden');save('position_value')
    corr=test[['age','minutes_played','goals_per90','assists_per90','yellow_per90','market_value_in_eur']].corr(method='spearman')
    plt.figure(figsize=(9,7));plt.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');plt.colorbar(label='Spearman rank correlation')
    labels=['Age','Minutes','Goals / 90','Assists / 90','Yellows / 90','Market value']
    plt.xticks(range(6),labels,rotation=40,ha='right');plt.yticks(range(6),labels)
    for i in range(6):
        for j in range(6):plt.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(corr.iloc[i,j])>.65 else 'black')
    plt.title('Performance and value · rank associations');save('correlations')
    plt.figure();plt.scatter(test.market_value_in_eur/1e6,test.prediction_eur/1e6,s=14,alpha=.35)
    lo=max(min(test.market_value_in_eur.min(),test.prediction_eur.min())/1e6,.001)
    hi=max(test.market_value_in_eur.max(),test.prediction_eur.max())/1e6
    plt.plot([lo,hi],[lo,hi],linestyle='--',color=COLORS[1],label='Perfect prediction')
    plt.xscale('log');plt.yscale('log');plt.xlabel('Observed market estimate (€ million, log scale)');plt.ylabel('Predicted market estimate (€ million, log scale)')
    plt.legend();plt.title('Untouched 2025 test: predicted versus observed');save('predictions')
    plt.figure();plt.scatter(test.prediction_eur/1e6,(test.prediction_eur-test.market_value_in_eur)/1e6,s=14,alpha=.35)
    plt.axhline(0,color=COLORS[1],linestyle='--');plt.xscale('log');plt.xlabel('Prediction (€ million, log scale)');plt.ylabel('Prediction minus observation (€ million)');plt.title('Residuals reveal errors on expensive players');save('residuals')
    plt.figure();top=imp.head(10).iloc[::-1];plt.barh(top.feature,top.mae_increase_eur/1e6,xerr=top.std_eur/1e6,color=COLORS[0])
    plt.xlabel('Increase in MAE when shuffled (€ million; mean ± SD)');plt.ylabel('Original input feature');plt.title('Permutation importance · 2024 calibration cohort')
    save('importance','Source: Transfermarkt via dcaribou · 2024 calibration set; five permutations, fixed selected model')
    ks=m['clustering']['silhouette_by_k'];plt.figure();plt.plot([int(k) for k in ks],list(ks.values()),marker='o')
    plt.xlabel('Number of clusters (k)');plt.ylabel('Silhouette score (higher is better)');plt.title('Select profile count using historical training data')
    save('silhouette','Source: Transfermarkt via dcaribou · 2018–2023 training cohorts; sample ≤3,000 player-seasons')
    plt.figure();
    for label,group in test.groupby('cluster'):
        plt.scatter(group.pc1,group.pc2,s=14,alpha=.4,label=f'Profile {label} (n={len(group)})',color=COLORS[int(label)])
    plt.legend(fontsize=8);plt.xlabel(f'Principal component 1 ({m["clustering"]["pca_explained_variance"][0]:.0%} variance)')
    plt.ylabel(f'Principal component 2 ({m["clustering"]["pca_explained_variance"][1]:.0%} variance)');plt.title('Automatically assigned player profiles · PCA projection');save('profiles')
    values=clusters[PROFILE_FEATURES];standard=(values-values.mean())/values.std(ddof=0).replace(0,1)
    plt.figure(figsize=(10,5));plt.imshow(standard,aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2);plt.colorbar(label='Standardized cluster mean (across cluster means)')
    plt.xticks(range(len(PROFILE_FEATURES)),['Minutes','Goals / 90','Assists / 90','Yellows / 90','Minutes / appearance'],rotation=20,ha='right')
    plt.yticks(range(len(clusters)),[f'Profile {c}' for c in clusters.cluster]);plt.xlabel('Performance indicator');plt.ylabel('Assigned profile');plt.title('What distinguishes the performance profiles?');save('cluster_heatmap')
    write_readme(m,audit,test,imp,clusters,tx)
    write_html(m,test)
    write_notebook()
    versions={name:importlib.metadata.version(name) for name in ['pandas','numpy','scikit-learn','scipy','matplotlib','requests','beautifulsoup4','joblib','pytest']}
    Path('requirements-lock.txt').write_text('\n'.join(f'{name}=={version}' for name,version in versions.items())+'\n')
    Path('reports/environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':versions},indent=2))
    print('README, 11 figure pairs, HTML explorer and notebook generated.',flush=True)

def write_readme(m,audit,test,imp,clusters,tx):
    score=m['test_2025']['selected']
    metrics_frame=pd.DataFrame([{'Model / cohort':name,'N':r['n'],'MAE (€m)':f'{r["mae_eur"]/1e6:.3f}',
        'RMSE (€m)':f'{r["rmse_eur"]/1e6:.3f}','R²':f'{r["r2"]:.3f}', 'Median APE':f'{r["median_ape_percent"]:.1f}%'}
        for name,r in m['test_2025'].items()])
    validation=pd.DataFrame([{'Model':name,'2023 MAE (€m)':f'{r["mae_eur"]/1e6:.3f}',
        'Temporal CV MAE (€m)':f'{m["search"][name]["cv_mae_eur"]/1e6:.3f}',
        'Best parameters':json.dumps(m['search'][name]['best_parameters'])} for name,r in m['validation_2023'].items()])
    split=pd.DataFrame([{'Role':role,'Years':years,'Rows':rows} for role,years,rows in [
        ('Hyperparameter training','2018–2022',m['split_rows']['train']),('Model selection','2023',m['split_rows']['validation']),
        ('Interval calibration','2024',m['split_rows']['calibration']),('Final test','2025',m['split_rows']['test'])]])
    example=test.sort_values('player_id').head(5)[['name','position','market_value_in_eur','prediction_eur','lower90_eur','upper90_eur','cluster']].copy()
    for c in ['market_value_in_eur','prediction_eur','lower90_eur','upper90_eur']:
        example[c]=example[c].map(lambda x:f'€{x/1e6:.2f}m')
    description=[]
    for _,r in clusters.iterrows():
        description.append(f'- **Profile {int(r.cluster)}**: {int(r.players)} test players; {r.minutes_played:.0f} minutes, {r.goals_per90:.2f} goals/90, {r.assists_per90:.2f} assists/90, {r.yellow_per90:.2f} yellows/90 and {r.minutes_per_appearance:.1f} minutes/appearance on average.')
    persistence=m['test_2025']['persistence']['mae_eur']
    conclusion=('The performance-only model improves on persistence on the matched subset.' if m['test_2025']['selected_on_persistence_subset']['mae_eur']<persistence else 'The last-known-value persistence baseline is stronger than the performance-only model on the matched subset. This is a useful negative result: these limited performance statistics do not replace prior market information.')
    raw_table = table(pd.DataFrame([{'Table':k,'Loaded rows':f'{v:,}'} for k,v in audit['raw_rows'].items()]))
    fee_table = table(pd.DataFrame([{'Indicator':name,'Spearman correlation with fee':f'{value:.3f}'} for name,value in tx['spearman_with_fee'].items()]))
    text=f'''# Football Player Market Value Prediction

**Analyse des performances et de la valeur marchande des joueurs de football**

A complete reproducible study of real Transfermarkt-derived data: collection, preprocessing, exploratory analysis, temporal regression, hyperparameter optimization, player segmentation, uncertainty assessment and reported transfer-fee analysis.

**Executed results:** {audit['rows']:,} player-season records, {audit['players']:,} distinct players, 2018–2025 cutoffs. Selected estimator: **{m['selected_model']}**. Final 2025 test MAE **€{score['mae_eur']/1e6:.2f}m**, RMSE **€{score['rmse_eur']/1e6:.2f}m**, R² **{score['r2']:.3f}**. These are measured holdout results, not illustrative numbers.

{conclusion}

![2025 observed versus predicted values](reports/figures/predictions.png)

## Contents

1. [Objectives and scope](#objectives-and-scope)
2. [Data and provenance](#data-and-provenance)
3. [Run the project](#run-the-project)
4. [Data schema and preprocessing](#data-schema-and-preprocessing)
5. [Exploratory analysis](#exploratory-analysis)
6. [Regression and optimization](#regression-and-optimization)
7. [Measured results](#measured-results)
8. [Player segmentation](#player-segmentation)
9. [Actual transfer fees](#actual-transfer-fees)
10. [Prediction examples](#prediction-examples)
11. [Limits and interpretation](#limits-and-interpretation)

## Objectives and scope

- Collect player information originating from Transfermarkt, with a published historical snapshot and an optional direct web scraper.
- Study individual performance: appearances, minutes, goals, assists, cards and exposure-adjusted rates.
- Predict the first positive market estimate recorded in the 92-day window beginning 1 May, using only league matches in the preceding 365 days. The May cutoff precedes the main end-of-season valuation updates; choosing a July cutoff would exclude most routinely updated players.
- Discover performance profiles using unsupervised learning, without including prices in the clustering input.
- Examine reported paid transfer fees separately from estimated market values.

The five leagues are Premier League (`GB1`), La Liga (`ES1`), Bundesliga (`L1`), Serie A (`IT1`) and Ligue 1 (`FR1`). Each observation is one player at one annual cutoff. This is a short-horizon summer valuation task, not a long-horizon forecast and not a transfer-fee prediction model.

## Data and provenance

Source: [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), derived from [Transfermarkt](https://www.transfermarkt.com/). The upstream dataset is distributed under **CC0-1.0**. See [data licensing](DATA_LICENSE.md) and [source manifest](data/source_manifest.json) for URLs, exact download times, byte counts and SHA256 checksums.

The upstream maintainer reports that updates paused in July 2026. We use historical years ending in 2025; no claim is made that these are October 2026 player values. Observed maximum dates in the downloaded files are **{audit['raw_max_dates']['appearances']}** for appearances and **{audit['raw_max_dates']['valuations']}** for valuations.

{raw_table}

The primary collection mode downloads published scraped CSV files rather than repeatedly requesting thousands of web pages. `src/scrape.py` also implements direct HTML extraction with BeautifulSoup, a robots.txt check, request spacing, caching, explicit currency parsing and fail-fast handling of blocks or layout changes. See [collection execution notes](reports/collection_notes.md) for the live attempt. The direct scrape is a separate current-data demonstration and is never mixed into historical training features.

Raw files are excluded from Git; the processed research cohort, fitted model artifacts and measured outputs are included. The downloader rejects a checksum change against the committed manifest unless `--refresh` is explicitly requested. Identical reruns depend on the same upstream bytes; if the source changes, obtain the hash-matched snapshot or deliberately regenerate the study.

## Run the project

Python 3.12 is recommended. Run all commands from the repository root.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m src.run
python -m pytest -q
```

`requirements-lock.txt` records direct package versions used for this run; transitive dependencies are resolved by pip and are not fully locked. `requirements.txt` offers broader supported ranges. [Environment metadata](reports/environment.json) records the execution environment. A full run downloads roughly 60 MB of compressed data, requires several hundred MB of RAM and trains multiple estimators; duration depends on hardware.

For an intentional data update:

```bash
python -m src.collect --refresh
python -m src.run --skip-download
```

To retrain directly from the committed processed cohort without downloading raw tables:

```bash
python -m src.train
python -m src.predict --input examples/player.csv --output predictions.csv
```

`src.report` additionally needs raw transfer and valuation tables for the fee study. Open [the notebook](notebooks/01_analysis.ipynb) in Jupyter, or [the standalone results explorer](reports/explorer.html) in a browser. The notebook requires Jupyter separately; the core pipeline does not.

Optional single-club scrape (subject to the site's access rules):

```bash
python -m src.scrape "https://www.transfermarkt.com/manchester-city/kader/verein/281"
```

## Data schema and preprocessing

```mermaid
erDiagram
    PLAYERS ||--o{{ APPEARANCES : player_id
    PLAYERS ||--o{{ PLAYER_VALUATIONS : player_id
    PLAYERS ||--o{{ TRANSFERS : player_id
    PLAYERS ||--o{{ PLAYER_SEASONS : player_id
    PLAYERS {{
        int player_id PK
        string name
        date date_of_birth
        string position
        float height_in_cm
    }}
    APPEARANCES {{
        int game_id
        int player_id FK
        date date
        string competition_id
        int minutes_played
        int goals
        int assists
    }}
    PLAYER_VALUATIONS {{
        int player_id FK
        date date
        float market_value_in_eur
    }}
    PLAYER_SEASONS {{
        int player_id FK
        int year
        date cutoff
        date target_date
        float market_value_in_eur
    }}
    TRANSFERS {{
        int player_id FK
        date transfer_date
        float transfer_fee
    }}
```

1. Parse all event dates; deduplicate appearances by `(game_id, player_id)` and valuations by `(player_id, date)`.
2. At each 1 May cutoff from 2018 to 2025, retain the preceding 365 days of five-league domestic appearances. Cup, international and other league games are excluded.
3. Require at least **450 minutes** of observed domestic play. A player who changed leagues retains combined five-league totals; league is the most recently observed league.
4. Aggregate appearances, minutes, goals, assists, yellow and red cards. Compute age at the cutoff, minutes per appearance and `count × 90 / minutes` rates. Keep plausible ages 15–45.
5. Select the earliest positive valuation in `[cutoff, cutoff + 92 days)`. Players without this target are omitted, creating a selected coverage cohort rather than a complete census.
6. Retain the last strictly pre-cutoff value only for the persistence benchmark; it is never a regression input.
7. Fit median numeric imputation, categorical mode imputation, standardization and one-hot encoding inside each training fold. Unknown categories are ignored at prediction time. Target values are modelled using `log1p`, then inverted to EUR and clipped at zero for reporting.

**Regression inputs (15):** {', '.join(FEATURES)}. Names, player IDs, current club, current value, peak historical value, contract end date, international caps and future events are excluded. [Complete data audit](reports/data_audit.json) includes missingness; [processed data](data/processed/player_seasons.csv) retains event dates for inspection.

```mermaid
flowchart LR
    A[Transfermarkt-derived CSVs] --> B[Date filtering and joins]
    B --> C[Annual player features]
    C --> D[2018–2022 expanding CV]
    D --> E[2023 estimator selection]
    E --> F[Refit through 2023]
    F --> G[2024 interval calibration]
    G --> H[Untouched 2025 evaluation]
    C --> I[Standardized performance features]
    I --> J[KMeans and silhouette selection]
    J --> K[2025 profiles and PCA]
```

## Exploratory analysis

![Value distribution](reports/figures/value_distribution.png)
![Age and value](reports/figures/age_value.png)
![Positions and values](reports/figures/position_value.png)
![Performance correlations](reports/figures/correlations.png)

The long upper tail motivates a log-transformed target. Age, playing exposure and attacking output can be associated with value, but these plots do not establish that changing one indicator causes a price change. Per-90 rates reduce differences in playing exposure; the minimum-minutes threshold limits some small-sample instability. The cohort's range and missingness are reported rather than filled before splitting. The 2020 cohort is much smaller because the pandemic disrupted match and valuation calendars; annual counts are available in the data audit. This makes the earliest validation fold less representative.

## Regression and optimization

Three estimator families are compared: regularized linear regression (Ridge), Random Forest and histogram gradient boosting. There are **11 hyperparameter combinations**, each evaluated on **three expanding temporal folds**: 2018–2019 → 2020, 2018–2020 → 2021 and 2018–2021 → 2022. Search scoring is negative MAE in original EUR, even though fitting uses the log target. Forests use 160 trees; boosting uses 180 iterations with internal random early stopping disabled.

{table(validation)}

The best candidate from each family is evaluated on 2023. The family with the smallest 2023 EUR MAE is selected, then refitted on 2018–2023. Neither calibration labels nor test labels select the estimator. Full fold scores are available in `reports/search_*.csv`.

{table(split)}

Recurring players may occur in multiple years. This reflects forecasting later values for a population containing incumbents, not an independent-player random split. The unseen-player test subset is also reported. Names and IDs are never estimator features.

## Measured results

{table(metrics_frame)}

`selected_on_persistence_subset` and `persistence` use the exact same rows. The median baseline uses the training-through-2023 median. RMSE emphasizes expensive-player errors; R² compares squared error against test-mean variation; median APE is the median absolute percentage error and is computed only on strictly positive targets. Log MAE is also preserved in [metrics JSON](reports/metrics.json).

![Residuals](reports/figures/residuals.png)
![Feature importance](reports/figures/importance.png)

The largest measured permutation effects are **{', '.join(imp.head(3).feature)}**. Importance is measured on 2024 by shuffling each original feature five times and reporting the increase in EUR MAE. Correlated inputs share information, so these scores are not causal effects or reliable evidence that a low-ranked variable is irrelevant.

Absolute log-residuals on 2024 calibrate a nominal 90% split-conformal-style interval. Empirical coverage on 2025 is **{m['interval']['test_coverage']:.1%}**, with median width **€{m['interval']['median_width_eur']/1e6:.2f}m**. Temporal drift and repeated players violate simple exchangeability assumptions; the nominal level is not a guaranteed future coverage rate. The model is not retrained on calibration data after interval construction.

## Player segmentation

KMeans uses standardized **{', '.join(PROFILE_FEATURES)}** only. We select `k` from 2–6 by the largest silhouette score on up to 3,000 sampled training player-seasons, with 20 initializations and seed 42. The fitted historical transformation and centroids assign profiles to 2025 players. **Selected k = {m['clustering']['k']}**.

![Silhouette selection](reports/figures/silhouette.png)
![Player profiles](reports/figures/profiles.png)
![Cluster characteristics](reports/figures/cluster_heatmap.png)

{chr(10).join(description)}

The selected silhouette is modest (about 0.25), indicating overlapping rather than sharply separated profiles. These numeric labels are arbitrary descriptive groups, not scouting grades. Price and age means in the [cluster summary](reports/cluster_summary.csv) describe the assigned groups after fitting and do not drive clustering. PCA is a 2D display only; clustering is performed in the full five-dimensional standardized feature space. Identical player-seasons across years contribute separately, so frequent incumbents have more influence. Defensive ability and goalkeeper quality cannot be identified well from these limited metrics.

## Actual transfer fees

The separate fee study matches **{tx['paid_transfers']} paid transactions** from July–September 2025 to the five-league 2025 feature cohort and the last positive valuation **strictly before each transaction**. Free moves, missing fees and nonpositive reported fees are excluded. Players may have multiple distinct transactions. Median fee is **€{tx['median_fee_eur']/1e6:.2f}m**; median fee/pre-transfer-value ratio is **{tx['median_fee_to_prior_value_ratio']:.2f}**.

![Actual transfer fees](reports/figures/transfer_fees.png)

{fee_table}

These are associations for disclosed positive-fee transactions, not a causal price equation. Negotiating power, contract length, loan structure, bonuses, injury history, wages and club finances are not controlled. Reported fees can be incomplete. A Transfermarkt market estimate is not the same variable as a realized transfer fee; the regression above predicts only the former. See [transaction rows](reports/transfer_analysis.csv) and [fee metrics](reports/transfer_results.json).

## Prediction examples

The following are actual held-out 2025 observations. They illustrate model errors and interval width, not today's market prices.

{table(example)}

```bash
python -m src.train
python -m src.predict --input examples/player.csv --output predictions.csv
```

The example CSV contains one real held-out player's prepared features. To predict another player, supply all 15 named columns with the same units and 365-day league-only aggregation. Numeric nulls are imputed; missing column names are rejected. Derived per-90 rates must be internally consistent. Outputs include `prediction_eur`, `lower90_eur`, `upper90_eur` and `cluster`. Fitted model artifacts are included for the recorded package versions and can be recreated locally by training; only load trusted joblib files.

## Repository map

```text
src/collect.py        Published snapshot downloader and checksums
src/scrape.py         Optional cached HTML squad scraper
src/features.py       Dated cohort construction and data audit
src/train.py          Temporal model search, intervals and clustering
src/predict.py        Batch valuation and profile inference
src/report.py         Figures, README, notebook and HTML explorer
src/run.py            End-to-end entry point
data/processed/      Reusable annual research cohort
data/source_manifest.json  Source URLs, dates and hashes
reports/              Measured results, predictions, fee study and figures
notebooks/            Exploratory analysis notebook
examples/             Real prepared input and generated prediction
tests/                Date, leakage, parser and metric checks
.github/workflows/    Automated tests on push and pull request
```

## Limits and interpretation

- **Point-in-time limitation:** position, foot and height come from the downloaded player profile snapshot, not archived annual profiles. Position changes and corrected metadata can introduce look-ahead bias. Match events, ages and valuation labels are dated, but the whole dataset should not be described as perfectly point-in-time.
- **Coverage bias:** only players with enough minutes and a recorded summer value enter the cohort. Absent players are not assigned zero value. Results do not generalize automatically to lower leagues, women's football, academy players or low-minute substitutes.
- **Prediction horizon:** the target may occur from the cutoff to 91 days later; per-row target dates are retained. Near-cutoff valuations may be almost unchanged, which explains why persistence can be strong.
- **Missing quality signals:** no xG, xA, tackles, saves, passing progression, injury or wage data. Goals and assists alone are incomplete performance measures, especially outside attacking positions.
- **Uncertainty:** a single temporal test period and global residual intervals do not establish stable performance across seasons, positions, leagues or price levels. No causal conclusions are claimed.
- **Performance versus price:** this project studies relationships between observed metrics and market estimates; it does not estimate causal determinants of on-field ability. The actual-fee section provides a separate observational analysis.
- **Operational suitability:** the persistence comparison must be considered before deployment. The research estimator demonstrates the full data-science workflow and should not be used as an authoritative transfer valuation.
- **Reproducibility:** seeds are fixed at 42 and direct package versions are recorded. Changes in upstream data, dependencies or numerical libraries can change results. Raw snapshot hashes make such changes visible.

## Verification

Run `python -m pytest -q`. Tests cover cutoff boundaries, earliest summer target selection, separation of previous values from model inputs, currency units, parser failure on block pages, known metric values and integrity of the committed processed cohort. GitHub Actions runs the same tests. The committed [execution record](reports/verification.json) contains local verification outcomes; a local pass does not assert that remote CI has passed.

## License and references

Project source code: [MIT](LICENSE). Data: upstream [CC0-1.0](https://github.com/dcaribou/transfermarkt-datasets/blob/master/LICENSE), with attribution and snapshot details in [DATA_LICENSE.md](DATA_LICENSE.md). No affiliation with Transfermarkt or Sofascore is implied.

- [Upstream dataset and update status](https://github.com/dcaribou/transfermarkt-datasets)
- [Upstream web scraper](https://github.com/dcaribou/transfermarkt-scraper)
- [scikit-learn model selection](https://scikit-learn.org/stable/modules/grid_search.html)
- [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
'''
    Path('README.md').write_text(text,encoding='utf-8')
    Path('examples').mkdir(exist_ok=True)
    test.sort_values('player_id').head(1)[['name']+FEATURES].to_csv('examples/player.csv',index=False)

def write_html(m,test):
    rows=test[['name','position','league','age','market_value_in_eur','prediction_eur','cluster']].to_dict('records')
    payload=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Football market value research</title>
<style>body{font:16px system-ui;margin:40px auto;padding:0 24px;max-width:1200px;color:#183b56;background:#f7f8fa}h1{font-size:32px}small{color:#526574}.stats{display:flex;gap:32px;flex-wrap:wrap;margin:28px 0}.stats b{display:block;font-size:26px}input,select{padding:10px;border:1px solid #bcc8cf;border-radius:4px;margin:6px}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #dae1e6}th{cursor:pointer}img{width:100%;max-width:750px}.wrap{overflow-x:auto}a{color:#166c80}</style>
<h1>Football Player Market Value Prediction</h1><p>Measured 2025 holdout results · five European leagues · historical Transfermarkt data</p>
<div class="stats">STATS</div><p>Market estimates are not actual transfer fees. These are historical observations, not current prices.</p>
<p><a href="../README.md">Full methodology and limitations</a> · <a href="metrics.json">Machine-readable results</a></p>
<img src="figures/predictions.png" alt="Predicted versus observed market value on log scales">
<h2>Explore real held-out players</h2><label>Search name <input id="q" placeholder="Player name"></label><label>Position <select id="position"><option value="">All</option></select></label><label>Profile <select id="profile"><option value="">All</option></select></label>
<p id="count"></p><div class="wrap"><table><thead><tr><th data-key="name">Player ↕</th><th data-key="position">Position ↕</th><th data-key="league">League ↕</th><th data-key="age">Age ↕</th><th data-key="market_value_in_eur">Observed (€m) ↕</th><th data-key="prediction_eur">Predicted (€m) ↕</th><th data-key="cluster">Profile ↕</th></tr></thead><tbody id="rows"></tbody></table></div>
<p><small>Values observed May–July 2025. Click column headers to sort. Search and filters work locally; no external requests. Display is limited to 100 matching rows.</small></p>
<h2>Performance profiles</h2><img src="figures/cluster_heatmap.png" alt="Standardized performance characteristics by cluster"><h2>Actual paid transfers</h2><img src="figures/transfer_fees.png" alt="Paid transfer fees versus strictly preceding estimates">
<script>const data=PAYLOAD;let key='market_value_in_eur',dir=-1;const q=document.getElementById('q'),p=document.getElementById('position'),c=document.getElementById('profile');for(const v of [...new Set(data.map(r=>r.position))].sort()){let o=new Option(v,v);p.add(o)}for(const v of [...new Set(data.map(r=>r.cluster))].sort()){c.add(new Option(v,v))}function render(){let rows=data.filter(r=>r.name.toLowerCase().includes(q.value.toLowerCase())&&(!p.value||r.position===p.value)&&(c.value===''||String(r.cluster)===c.value));rows.sort((a,b)=>dir*(typeof a[key]==='string'?a[key].localeCompare(b[key]):a[key]-b[key]));document.getElementById('count').textContent=rows.length+' matching players';let body=document.getElementById('rows');body.replaceChildren();for(const r of rows.slice(0,100)){let tr=document.createElement('tr');for(const v of [r.name,r.position,r.league,r.age.toFixed(1),(r.market_value_in_eur/1e6).toFixed(2),(r.prediction_eur/1e6).toFixed(2),r.cluster]){let td=document.createElement('td');td.textContent=v;tr.append(td)}body.append(tr)}}for(const el of [q,p,c])el.addEventListener('input',render);document.querySelectorAll('th[data-key]').forEach(th=>th.addEventListener('click',()=>{dir=key===th.dataset.key?-dir:-1;key=th.dataset.key;render()}));render();</script></html>'''
    stats=f'<div><b>{len(test):,}</b>Test players</div><div><b>€{m["test_2025"]["selected"]["mae_eur"]/1e6:.2f}m</b>Test MAE</div><div><b>{m["test_2025"]["selected"]["r2"]:.3f}</b>Test R²</div><div><b>{m["clustering"]["k"]}</b>Profiles</div>'
    Path('reports/explorer.html').write_text(text.replace('STATS',stats).replace('PAYLOAD',payload),encoding='utf-8')

def write_notebook():
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(keepends=True)})
    md('# Analyse des performances et de la valeur marchande\nReal Transfermarkt-derived historical study. Run the pipeline first; this notebook inspects committed outputs. All prices are EUR. No synthetic player data is used for the study.')
    code("from pathlib import Path\nimport os, json\nif Path.cwd().name == 'notebooks': os.chdir('..')\nimport pandas as pd\nfrom IPython.display import display, Image\ndata = pd.read_csv('data/processed/player_seasons.csv')\nmetrics = json.loads(Path('reports/metrics.json').read_text())\ndisplay(data.head())\ndisplay(data.groupby('year').size().rename('player_seasons'))")
    md('## Qualité et sélection\nOnly players with at least 450 minutes and a recorded summer valuation are included. Snapshot positions may introduce look-ahead bias. Inspect the audit and date boundaries before interpreting results.')
    code("display(pd.Series(json.loads(Path('reports/data_audit.json').read_text())['missing_percent'], name='missing_percent'))\nassert (pd.to_datetime(data.last_match_date) < pd.to_datetime(data.cutoff)).all()\nassert (pd.to_datetime(data.target_date) >= pd.to_datetime(data.cutoff)).all()\ndisplay(Image('reports/figures/correlations.png'))")
    md('## Modèles et validation temporelle\nHyperparameters use expanding folds through 2022; 2023 selects the estimator, 2024 calibrates intervals, 2025 is the final test. Compare persistence on the matched subset.')
    code("display(pd.DataFrame(metrics['test_2025']).T)\ndisplay(Image('reports/figures/predictions.png'))\ndisplay(pd.read_csv('reports/importance.csv'))")
    md('## Profils automatiques\nKMeans uses five performance indicators and excludes prices. PCA is only a projection; cluster labels are arbitrary.')
    code("display(pd.read_csv('reports/cluster_summary.csv'))\ndisplay(Image('reports/figures/profiles.png'))")
    md('## Prix réels des transferts\nReported paid fees and strictly pre-transfer estimates are separate variables. Associations are not causal; free and undisclosed moves are excluded.')
    code("display(pd.Series(json.loads(Path('reports/transfer_results.json').read_text())))\ndisplay(Image('reports/figures/transfer_fees.png'))")
    md('## Exemple de prédiction\nTrain locally to create trusted model artifacts: `python -m src.train`. Then run the following cell.')
    code("from src.predict import predict\nif Path('artifacts/valuation.joblib').exists():\n    display(predict(pd.read_csv('examples/player.csv')))\nelse:\n    print('Run python -m src.train to create the local models.')")
    Path('notebooks').mkdir(exist_ok=True)
    Path('notebooks/01_analysis.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},indent=2),encoding='utf-8')

if __name__ == '__main__':
    report()
