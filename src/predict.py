"""Batch inference for prepared rows (EUR predictions and descriptive profiles)."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def predict(frame, model_dir='artifacts'):
    bundle = joblib.load(Path(model_dir)/'valuation.joblib')
    missing = set(bundle['features'])-set(frame.columns)
    if missing:
        raise ValueError(f'Missing features: {sorted(missing)}')
    result = frame.copy()
    value = np.maximum(bundle['model'].predict(frame[bundle['features']]),0)
    result['prediction_eur'] = value
    result['lower90_eur'] = np.maximum(np.expm1(np.log1p(value)-bundle['log_interval_radius']),0)
    result['upper90_eur'] = np.expm1(np.log1p(value)+bundle['log_interval_radius'])
    profile = joblib.load(Path(model_dir)/'profiles.joblib')
    result['cluster'] = profile['cluster'].predict(profile['preprocessing'].transform(frame[profile['features']]))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',default='predictions.csv')
    parser.add_argument('--models',default='artifacts')
    args=parser.parse_args()
    result=predict(pd.read_csv(args.input),args.models)
    result.to_csv(args.output,index=False)
    print(result[['prediction_eur','lower90_eur','upper90_eur','cluster']].to_string(index=False))
