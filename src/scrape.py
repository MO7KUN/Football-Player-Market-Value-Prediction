"""Optional polite Transfermarkt squad scraper, separate from historical modelling data."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

def parse_value(text):
    """Transfermarkt English currency strings -> EUR; missing stays missing."""
    text = text.strip().replace('€','').replace(',','')
    if text in ('', '-', 'N/A'):
        return None
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(bn|m|k)?', text, re.I)
    if not match:
        raise ValueError(f'Unknown market value format: {text!r}')
    return float(match[1]) * {None:1, 'k':1e3, 'm':1e6, 'bn':1e9}[match[2].lower() if match[2] else None]

def parse_squad(html):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.select_one('table.items')
    if table is None:
        raise ValueError('No squad table: blocked response or changed page structure')
    rows = []
    for row in table.select('tbody > tr'):
        link = row.select_one('td.hauptlink a[href*="/profil/spieler/"]')
        value = row.select_one('td.rechts.hauptlink')
        if link is None or value is None:
            continue
        match = re.search(r'/spieler/(\d+)',link.get('href',''))
        if match:
            rows.append({'player_id':int(match[1]), 'name':link.get_text(strip=True),
                         'market_value_in_eur':parse_value(value.get_text(strip=True))})
    if not rows:
        raise ValueError('No player records parsed; refusing silent empty output')
    return rows

def scrape(url, cache=Path('data/raw/html')):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ('www.transfermarkt.com', 'www.transfermarkt.co.uk'):
        raise ValueError('Only HTTPS Transfermarkt English squad URLs are supported')
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (hashlib.sha256(url.encode()).hexdigest()+'.html')
    if not path.exists():
        session = requests.Session()
        session.headers['User-Agent'] = 'FootballValueResearch/1.0 (educational; one cached squad request)'
        robots = session.get(f'{parsed.scheme}://{parsed.netloc}/robots.txt',timeout=30)
        robots.raise_for_status()
        rules = RobotFileParser()
        rules.parse(robots.text.splitlines())
        if not rules.can_fetch('FootballValueResearch',url):
            raise PermissionError('robots.txt disallows this URL; use the published historical snapshot')
        time.sleep(max(3,rules.crawl_delay('FootballValueResearch') or 0))
        response = session.get(url,timeout=45)
        response.raise_for_status()  # Stop on 403/429; no bypass or rapid retry.
        parse_squad(response.text)
        path.write_text(response.text,encoding='utf-8')
    return parse_squad(path.read_text(encoding='utf-8'))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--output',default='data/raw/scraped_squad.json')
    args=parser.parse_args()
    rows=scrape(args.url)
    Path(args.output).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Collected {len(rows)} players into {args.output}')
