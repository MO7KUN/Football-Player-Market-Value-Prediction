# Data attribution and licensing

The historical research data comes from [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), which publishes Transfermarkt-derived data under [CC0-1.0](https://github.com/dcaribou/transfermarkt-datasets/blob/master/LICENSE). The project code is MIT licensed separately.

The committed processed player-season cohort, test predictions and transfer-analysis extracts are transformations of that published snapshot. Source URLs, download timestamps and SHA256 hashes are in `data/source_manifest.json`. Raw compressed files are intentionally excluded from version control.

The live HTML scraper is an optional demonstration. Its current squad extraction is stored only locally in ignored `data/raw/scraped_squad.json`; the scraped website's own terms apply to that extraction. It is not relicensed here as CC0 and is not used in the historical model. The public execution notes record request metadata and row count, without redistributing cached HTML or the direct extraction.

Transfermarkt is the original data source. This project has no affiliation with Transfermarkt or Sofascore. Respect source access rules and attribution when collecting or reusing data.
