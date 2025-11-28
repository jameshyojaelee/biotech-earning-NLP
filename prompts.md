

### 6) Reproducibility and data version pinning
```
Harden reproducibility.
1) In config/config.yaml add hf_dataset_revision (commit hash) and price_cache_dir (e.g., data_raw/prices). Use these in ingest and returns modules.
2) Modify src/ingest/hf_ingest.py to load the dataset with the specified revision and log it.
3) Modify src/finance/returns.py to optionally read/write cached prices per ticker to price_cache_dir before hitting yfinance; add a --refresh-cache flag to compute_returns_for_events.py.
4) Update README.md with a short “Reproducibility” note explaining dataset revision and price caching.
5) Add tests (can be lightweight/mocked) to ensure cache paths are honored (mock yfinance.download to avoid network).
6) Run pytest -q.
Summarize config changes and caching behavior.
```

### 7) Streamlit polish (optional)
```
If time allows, add a couple of polish items:
1) In app.py, add selectbox to choose return window (abn_ret_1d vs abn_ret_5d) and display chosen metric.
2) Add a scatter plot of qa_sent_score vs chosen abnormal return for the selected ticker, with a trendline.
3) Add a download button to export the ticker’s rows as CSV.
4) Upgrade the overall theme with always dark-mode theme along with more trendy colors to make it look stylish
Keep changes backward-compatible.
```