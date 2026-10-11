import numpy as np
import pandas as pd
import pytest
from src.features import season_rows, FEATURES
from src.scrape import parse_value, parse_squad
from src.train import metrics

def test_features_respect_cutoff_and_choose_first_future_value():
    players=pd.DataFrame([{'player_id':1,'name':'Fixture','date_of_birth':pd.Timestamp('2000-01-01'),
                           'height_in_cm':180,'position':'Attack','foot':'right'}])
    a=pd.DataFrame([{'player_id':1,'game_id':i,'date':pd.Timestamp(date),'competition_id':'GB1',
                    'minutes_played':90,'goals':goals,'assists':0,'yellow_cards':0,'red_cards':0}
                   for i,date,goals in [(1,'2023-04-01',1),(2,'2023-05-01',99),(3,'2023-05-02',99)]])
    v=pd.DataFrame({'player_id':[1]*3,'date':pd.to_datetime(['2023-04-30','2023-05-05','2023-06-05']),
                    'market_value_in_eur':[1e6,2e6,9e6]})
    rows=season_rows(players,a,v,2023,min_minutes=90)
    assert len(rows)==1
    assert rows.iloc[0].goals==1
    assert rows.iloc[0].market_value_in_eur==2e6
    assert rows.iloc[0].prior_value_eur==1e6
    assert rows.iloc[0].goals_per90==1
    assert set(FEATURES).isdisjoint({'market_value_in_eur','prior_value_eur','player_id'})

@pytest.mark.parametrize('text,value',[('€1.5m',1500000),('€750k',750000),('€1.2bn',1200000000),('-',None)])
def test_currency(text,value):
    assert parse_value(text)==value

def test_scraper_rejects_block_page():
    with pytest.raises(ValueError):
        parse_squad('<html>Access denied</html>')

def test_scraper_fixture():
    html='<table class="items"><tbody><tr><td class="hauptlink"><a href="/a/profil/spieler/42">A</a></td><td class="rechts hauptlink">€2m</td></tr></tbody></table>'
    assert parse_squad(html)==[{'player_id':42,'name':'A','market_value_in_eur':2000000}]

def test_metrics_known_values():
    result=metrics([1e6,3e6],[1e6,1e6])
    assert result['mae_eur']==1e6
    assert result['rmse_eur']==pytest.approx(np.sqrt(2)*1e6)

def test_real_processed_data_temporal_integrity():
    from pathlib import Path
    path=Path('data/processed/player_seasons.csv')
    if not path.exists():
        pytest.skip('Run feature pipeline first')
    data=pd.read_csv(path,parse_dates=['last_match_date','cutoff','target_date','prior_value_date'])
    assert (data.last_match_date<data.cutoff).all()
    assert (data.target_date>=data.cutoff).all()
    assert not data.duplicated(['player_id','year']).any()
    assert (data.minutes_played>=450).all()
    assert (data.market_value_in_eur>0).all()
