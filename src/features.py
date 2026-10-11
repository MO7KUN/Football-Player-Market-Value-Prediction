"""Strictly dated player-season features; no current or peak market values."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

LEAGUES = ['GB1', 'ES1', 'L1', 'IT1', 'FR1']
NUMERIC = ['age', 'height_in_cm', 'appearances', 'minutes_played', 'goals', 'assists',
           'yellow_cards', 'red_cards', 'goals_per90', 'assists_per90', 'yellow_per90',
           'minutes_per_appearance']
CATEGORICAL = ['position', 'foot', 'league']
FEATURES = NUMERIC + CATEGORICAL
PROFILE_FEATURES = ['minutes_played', 'goals_per90', 'assists_per90', 'yellow_per90', 'minutes_per_appearance']

def season_rows(players, appearances, valuations, year, min_minutes=450):
    cutoff = pd.Timestamp(year=year, month=5, day=1)
    start = cutoff - pd.Timedelta(days=365)
    a = appearances.loc[(appearances.date >= start) & (appearances.date < cutoff)
                        & appearances.competition_id.isin(LEAGUES)].sort_values('date')
    sums = a.groupby('player_id').agg(appearances=('game_id', 'nunique'),
        minutes_played=('minutes_played', 'sum'), goals=('goals', 'sum'), assists=('assists', 'sum'),
        yellow_cards=('yellow_cards', 'sum'), red_cards=('red_cards', 'sum'),
        last_match_date=('date', 'max'), league=('competition_id', 'last'))
    sums = sums.loc[sums.minutes_played >= min_minutes].reset_index()
    # Earliest recorded summer valuation, strictly after all feature events.
    targets = valuations.loc[(valuations.date >= cutoff) &
        (valuations.date < cutoff + pd.Timedelta(days=92)) & (valuations.market_value_in_eur > 0)]
    targets = targets.sort_values('date').drop_duplicates('player_id', keep='first')
    targets = targets[['player_id', 'date', 'market_value_in_eur']].rename(columns={'date':'target_date'})
    prior = valuations.loc[(valuations.date < cutoff) & (valuations.market_value_in_eur > 0)]
    prior = prior.sort_values('date').drop_duplicates('player_id', keep='last')
    prior = prior[['player_id', 'date', 'market_value_in_eur']].rename(
        columns={'date':'prior_value_date', 'market_value_in_eur':'prior_value_eur'})
    static = players[['player_id', 'name', 'date_of_birth', 'height_in_cm', 'position', 'foot']]
    frame = sums.merge(static, on='player_id', validate='one_to_one').merge(targets, on='player_id', validate='one_to_one')
    frame = frame.merge(prior, on='player_id', how='left', validate='one_to_one')
    frame['age'] = (cutoff - frame.date_of_birth).dt.days / 365.25
    frame = frame.loc[frame.age.between(15, 45)].copy()
    for count, rate in [('goals', 'goals_per90'), ('assists', 'assists_per90'), ('yellow_cards', 'yellow_per90')]:
        frame[rate] = 90 * frame[count] / frame.minutes_played
    frame['minutes_per_appearance'] = frame.minutes_played / frame.appearances
    frame['cutoff'] = cutoff
    frame['year'] = year
    return frame

def build(raw=Path('data/raw'), output=Path('data/processed/player_seasons.csv')):
    raw, output = Path(raw), Path(output)
    players = pd.read_csv(raw / 'players.csv.gz', parse_dates=['date_of_birth'])
    appearances = pd.read_csv(raw / 'appearances.csv.gz', parse_dates=['date'])
    valuations = pd.read_csv(raw / 'player_valuations.csv.gz', parse_dates=['date'])
    appearances = appearances.drop_duplicates(['game_id', 'player_id'])
    valuations = valuations.drop_duplicates(['player_id', 'date'])
    frame = pd.concat([season_rows(players, appearances, valuations, y) for y in range(2018, 2026)], ignore_index=True)
    assert (frame.last_match_date < frame.cutoff).all()
    assert (frame.target_date >= frame.cutoff).all()
    assert (frame.prior_value_date.dropna() < frame.loc[frame.prior_value_date.notna(), 'cutoff']).all()
    assert not frame.duplicated(['player_id', 'year']).any()
    assert set(FEATURES).isdisjoint({'market_value_in_eur', 'prior_value_eur', 'player_id', 'name'})
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    audit = {'rows':len(frame), 'players':int(frame.player_id.nunique()),
             'rows_by_year':frame.groupby('year').size().to_dict(),
             'missing_percent':(100*frame[FEATURES].isna().mean()).round(3).to_dict(),
             'raw_rows':{'players':len(players), 'appearances':len(appearances), 'valuations':len(valuations)},
             'raw_max_dates':{'appearances':str(appearances.date.max().date()), 'valuations':str(valuations.date.max().date())}}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/data_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    return frame

if __name__ == '__main__':
    build()
