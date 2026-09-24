<project-root>/
│
├── data-sources/
│   └── source=<source-id>/
│       ├── agreements/agreement=<id>/<document-file>
│       └── dictionaries/schema=<id>/version=<id>/dictionary.json
│
├── test-scripts/
│   └── test-script=<id>/byod/upload=<id>/
│       ├── original.xlsx              ← or original.csv
│       └── dictionary.json
│
├── pipeline-assets/
│   └── pipeline=<id>/asset=<id>/
│       └── original.sas               ← or original.sql
│
├── test-runs/
│   └── test-script=<id>/pipeline=<id>/execution=<id>/
│       ├── stages/stage=<id>/part-*.parquet
│       ├── outputs/dataset=<id>/part-*.parquet
│       └── manifest.json
│
└── shared-runs/
    └── pipeline=<id>/execution=<id>/
        ├── stages/stage=<id>/part-*.parquet
        ├── outputs/dataset=<id>/part-*.parquet
        └── manifest.json

<project-root>/shared-runs/pipeline=100/execution=200/
    outputs/dataset=employee_output/
        part-00000.parquet
        part-00001.parquet