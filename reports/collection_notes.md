# Collection execution record

## Published historical snapshot

Four real Transfermarkt-derived tables were downloaded from the public distribution linked by dcaribou/transfermarkt-datasets: players, appearances, player_valuations and transfers. Full source URLs, actual download times, sizes and SHA256 checksums are recorded in `data/source_manifest.json`.

## Direct web scraping demonstration

- URL: https://www.transfermarkt.com/manchester-city/kader/verein/281
- Execution date (UTC): 2026-10-11
- Implementation: requests + BeautifulSoup (`src/scrape.py`).
- robots.txt retrieved successfully; the parser permitted the request for the declared research user agent.
- A delay of at least three seconds preceded the single squad-page request.
- Result: successfully parsed **24 players**, including player IDs, names and numeric EUR market values.
- Cached HTML and the live extraction remain in ignored local raw-data storage; they are excluded from Git publication and historical training.
- Parser unit tests separately verify currency units and rejection of block pages. No proxy, CAPTCHA bypass or aggressive retry is used.

## Cohort design

Source-date inspection showed routine end-of-season valuations concentrated in May and June. Annual feature cutoffs are therefore 1 May, with the first positive target in the next 92 days. This choice follows data coverage; estimator selection uses 2023 only, with 2024 reserved for interval calibration and 2025 for final evaluation. Performance observations on or after the cutoff are excluded.
