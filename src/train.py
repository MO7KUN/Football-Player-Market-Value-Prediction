"""Temporal regression, hyperparameter search, calibrated intervals and player profiles."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import NUMERIC, CATEGORICAL, FEATURES, PROFILE_FEATURES

SEED = 42

def metrics(y, pred):
    y, pred = np.asarray(y), np.maximum(np.asarray(pred), 0)
    return {'n':len(y), 'mae_eur':float(mean_absolute_error(y, pred)),
        'rmse_eur':float(np.sqrt(mean_squared_error(y, pred))), 'r2':float(r2_score(y, pred)),
        'log_mae':float(mean_absolute_error(np.log1p(y), np.log1p(pred))),
        'median_ape_percent':float(100*np.median(np.abs(y-pred)/y))}

def preprocessing():
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC),
        ('category', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                              ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL)])

def regressor(model):
    return TransformedTargetRegressor(regressor=Pipeline([('prep', preprocessing()), ('model', model)]),
                                     func=np.log1p, inverse_func=np.expm1)

def train(path='data/processed/player_seasons.csv'):
    Path('reports').mkdir(exist_ok=True)
    Path('artifacts').mkdir(exist_ok=True)
    frame = pd.read_csv(path)
    train = frame.loc[frame.year <= 2022].reset_index(drop=True)
    valid = frame.loc[frame.year == 2023].copy()
    calibration = frame.loc[frame.year == 2024].copy()
    test = frame.loc[frame.year == 2025].copy()
    X, y = train[FEATURES], train.market_value_in_eur
    # Expanding temporal folds, never shuffle player-season rows.
    folds = [(np.flatnonzero(train.year < year), np.flatnonzero(train.year == year)) for year in [2020, 2021, 2022]]
    configs = {
      'ridge': (Ridge(), {'regressor__model__alpha':[1, 10, 100]}),
      'random_forest': (RandomForestRegressor(n_estimators=160, random_state=SEED, n_jobs=2),
                        {'regressor__model__max_depth':[12, None], 'regressor__model__min_samples_leaf':[3, 10]}),
      'gradient_boosting': (HistGradientBoostingRegressor(max_iter=180, early_stopping=False, random_state=SEED),
                        {'regressor__model__max_leaf_nodes':[15, 31], 'regressor__model__l2_regularization':[1, 10]})}
    models, searches, validation = {}, {}, {}
    for name, (base, grid) in configs.items():
        print(f'Tuning {name}', flush=True)
        search = GridSearchCV(regressor(base), grid, cv=folds, scoring='neg_mean_absolute_error', n_jobs=1)
        search.fit(X, y)
        models[name] = search.best_estimator_
        searches[name] = {'best_parameters': search.best_params_, 'cv_mae_eur':-float(search.best_score_)}
        pd.DataFrame(search.cv_results_).to_csv(f'reports/search_{name}.csv', index=False)
        validation[name] = metrics(valid.market_value_in_eur, search.predict(valid[FEATURES]))
    selected = min(validation, key=lambda name:validation[name]['mae_eur'])
    print(f'Selected on 2023 validation: {selected}', flush=True)
    # Refit through validation year. 2024 remains untouched for interval calibration.
    fit = frame.loc[frame.year <= 2023]
    model = clone(models[selected]).fit(fit[FEATURES], fit.market_value_in_eur)
    pred = np.maximum(model.predict(test[FEATURES]), 0)
    cal_pred = np.maximum(model.predict(calibration[FEATURES]), 0)
    errors = np.abs(np.log1p(calibration.market_value_in_eur) - np.log1p(cal_pred))
    level = min(np.ceil((len(errors)+1)*0.90)/len(errors), 1)
    radius = float(np.quantile(errors, level, method='higher'))
    lower = np.maximum(np.expm1(np.log1p(pred)-radius), 0)
    upper = np.expm1(np.log1p(pred)+radius)
    median = float(fit.market_value_in_eur.median())
    known = test.prior_value_eur.notna()
    result = {'selected_model':selected, 'features':FEATURES, 'search':searches, 'validation_2023':validation,
        'test_2025':{'selected':metrics(test.market_value_in_eur, pred),
                     'median_baseline':metrics(test.market_value_in_eur, np.full(len(test), median)),
                     'persistence':metrics(test.loc[known,'market_value_in_eur'], test.loc[known,'prior_value_eur']),
                     'selected_on_persistence_subset':metrics(test.loc[known,'market_value_in_eur'], pred[known])},
        'interval':{'nominal_coverage':0.9, 'calibration_year':2024, 'log_radius':radius,
                    'test_coverage':float(np.mean((test.market_value_in_eur>=lower)&(test.market_value_in_eur<=upper))),
                    'median_width_eur':float(np.median(upper-lower))},
        'split_rows':{'train':len(train), 'validation':len(valid), 'calibration':len(calibration),'test':len(test)}}
    unseen = ~test.player_id.isin(fit.player_id)
    if unseen.sum() > 1:
        result['test_2025']['unseen_players'] = metrics(test.loc[unseen,'market_value_in_eur'], pred[unseen])
    test['prediction_eur'], test['lower90_eur'], test['upper90_eur'] = pred, lower, upper
    test.to_csv('reports/test_predictions.csv', index=False)
    segments = []
    for label, group in test.groupby('position'):
        segments.append({'position':label, **metrics(group.market_value_in_eur, group.prediction_eur)})
    pd.DataFrame(segments).to_csv('reports/errors_by_position.csv', index=False)
    # Explain on validation only; test remains for evaluation.
    imp = permutation_importance(model, calibration[FEATURES], calibration.market_value_in_eur,
        n_repeats=5, scoring='neg_mean_absolute_error', random_state=SEED, n_jobs=1)
    pd.DataFrame({'feature':FEATURES, 'mae_increase_eur':imp.importances_mean,
                  'std_eur':imp.importances_std}).sort_values('mae_increase_eur',ascending=False).to_csv('reports/importance.csv',index=False)
    joblib.dump({'model':model, 'features':FEATURES, 'log_interval_radius':radius,
                 'trained_through':2023, 'calibrated_on':2024}, 'artifacts/valuation.joblib',compress=3)
    # Segmentation uses performance only, never price, name, league or position.
    profile = Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    Z = profile.fit_transform(fit[PROFILE_FEATURES])
    scores, candidates = {}, {}
    for k in range(2,7):
        candidate = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
        scores[k] = float(silhouette_score(Z, candidate.labels_, sample_size=min(3000,len(Z)),random_state=SEED))
        candidates[k] = candidate
    k = max(scores,key=scores.get)
    cluster = candidates[k]
    test['cluster'] = cluster.predict(profile.transform(test[PROFILE_FEATURES]))
    pca = PCA(n_components=2,random_state=SEED).fit(Z)
    xy = pca.transform(profile.transform(test[PROFILE_FEATURES]))
    test['pc1'],test['pc2'] = xy[:,0],xy[:,1]
    test.to_csv('reports/player_profiles.csv',index=False)
    summary = test.groupby('cluster')[PROFILE_FEATURES+['age','market_value_in_eur']].mean()
    summary['players'] = test.groupby('cluster').size()
    summary.to_csv('reports/cluster_summary.csv')
    result['clustering'] = {'k':k,'silhouette_by_k':scores,'pca_explained_variance':pca.explained_variance_ratio_.tolist(),
        'features':PROFILE_FEATURES, 'test_counts':test.cluster.value_counts().sort_index().to_dict()}
    joblib.dump({'preprocessing':profile,'cluster':cluster,'pca':pca,'features':PROFILE_FEATURES},'artifacts/profiles.joblib',compress=3)
    Path('reports/metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__ == '__main__':
    train()
