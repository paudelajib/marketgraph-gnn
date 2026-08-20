# MarketGraph — GNN-based Cross-Sectional Stock Direction Prediction

## The one-sentence pitch

> Given ~100 US large-caps, build a **dynamic graph** where nodes are stocks and edges encode
> statistical + fundamental relationships, and train temporal GNNs to rank which stocks will
> outperform the cross-section tomorrow — evaluated with a leak-free walk-forward backtest and
> served through an interactive web app that visualizes the market as a living network.

## Why this framing (and not "predict the price")

| Naive version | What we do instead | Why |
|---|---|---|
| Predict tomorrow's price | Predict sign of *excess* return vs. cross-sectional median | Price levels are ~random walks; a model that predicts $101 after $100 looks great and knows nothing |
| Random train/test split | Walk-forward with embargo | Random splits leak the future. This is the #1 thing an interviewer probes |
| Report accuracy | Rank IC, precision@k, long-short PnL after costs | 51% accuracy can be very profitable; 55% can lose money |
| One static correlation graph | Rolling graphs rebuilt each period + learned adjacency | The market's correlation structure is regime-dependent |

**Honest expectation:** daily direction signal in liquid US large-caps is weak — rank IC of
0.02–0.05 is a *real* result. The project is judged on methodological rigor, not on returns.
Setting that expectation explicitly in the README is itself a hiring signal.

---

## Phase 0 — Environment & data layer
**Goal:** one command pulls clean daily OHLCV for the universe into local Parquet, incrementally.

- `data/universe.py` — ticker list (start from `~/Desktop/STOCKS/Top_50_US_stocks.csv`, expand to ~100),
  with sector/industry metadata (needed later for the sector graph).
- `data/ingest.py` — `yfinance` download → tidy long-format DataFrame `(date, ticker, open, high, low, close, adj_close, volume)`.
- Cache to `data/raw/prices.parquet`; re-running only fetches missing dates.
- Handle: splits/dividends (use adjusted close), delistings, tickers with short history, NaN days.
- Sanity notebook: coverage heatmap (ticker × date), returns distribution, outlier scan.

**You learn:** survivorship bias, adjustment conventions, why point-in-time data matters.

## Phase 1 — Features, labels, and leak-free splitting
**Goal:** a `(T, N, F)` tensor of features and a `(T, N)` label matrix, with a splitter you trust.

- Node features per stock per day: multi-horizon returns (1/5/21/63d), realized vol, volume z-score,
  RSI/MACD-style momentum, distance from moving averages, market-relative beta. **Cross-sectionally
  z-scored each day** — this is what makes the model comparable across stocks and regimes.
- Labels: `y = 1` if next-day return > cross-sectional median (binary), plus the continuous
  excess return for ranking losses.
- `TimeSeriesSplitter` with train / **embargo gap** / validation / test. Write a unit test that
  asserts no train timestamp ≥ any val timestamp.

**You learn:** feature normalization for panel data, look-ahead bias, purging & embargo.

## Phase 2 — Baselines (do NOT skip)
**Goal:** the number the GNN has to beat.

1. Always-predict-up
2. Momentum heuristic (rank by trailing 21d return)
3. Logistic regression / LightGBM on node features only — **no graph at all**
4. Per-stock GRU on the feature sequence — temporal, still no graph

If the GNN doesn't beat #3 and #4, the graph is adding nothing, and knowing that is the point.

**You learn:** the discipline of ablation. Every later claim is "vs. baseline".

## Phase 3 — Graph construction (the research core)
**Goal:** `build_graph(window) -> A (N×N)`, several variants, swappable by config.

- **Correlation graph** — rolling 60d return correlation, sparsified by top-k neighbors or
  threshold; optionally distance-correlation or partial correlation to kill the market-factor
  effect that makes everything correlate at 0.6.
- **Sector graph** — same GICS sector = edge. Cheap, strong, interpretable.
- **Lead–lag graph** — directed edges from cross-correlation at lag 1 (does XOM move CVX tomorrow?).
- **Learned adjacency** — node embeddings `E ∈ R^{N×d}`, `A = softmax(relu(E₁E₂ᵀ))`, trained
  end-to-end (the MTGNN / Graph-WaveNet trick). Then *visualize what it learned* — great demo material.

**You learn:** how graph topology encodes an inductive bias, and that most of the gain in "graph ML
for finance" comes from graph construction, not from the fancy layer.

## Phase 4 — Models, in increasing order
**Goal:** a clean `nn.Module` progression you can explain line by line.

1. **GCN from scratch** — `H' = σ(Â H W)` with `Â = D^{-1/2}(A+I)D^{-1/2}`. With N≈100 the adjacency
   is a dense 100×100 matrix, so this is *literally two matmuls* — write it in raw PyTorch rather
   than reaching for a library. You'll understand message passing permanently.
2. **GAT** — implement attention coefficients by hand; inspect which neighbors matter.
3. **Temporal + spatial** — GRU/TCN encoder per node, then graph layers across nodes each step
   (STGCN-style). This is the real model.
4. **Dynamic graph** — recompute (or re-learn) `A` per window; compare against frozen `A`.
5. Optional stretch: heterogeneous graph adding sector/index nodes.

Losses to compare: BCE (classification) vs. a **listwise ranking loss** (ListMLE / soft-IC), since
the actual objective is ordering, not calibration.

**You learn:** GNN internals without a black box; when temporal-then-spatial beats interleaved.

## Phase 5 — Evaluation & backtest harness
**Goal:** results you'd defend in an interview.

- Walk-forward: retrain every 6 months, roll forward, concatenate out-of-sample predictions.
- Metrics: accuracy, AUC, **rank IC + IC t-stat**, precision@10, turnover.
- Long-short backtest: long top decile / short bottom decile, daily rebalance,
  **with transaction costs** (5–10 bps). Report Sharpe, max drawdown, and the equity curve
  *with and without* costs — the gap is usually the whole story.
- Seed variance: 5 seeds, report mean ± std. Single-seed results are noise.
- Explainability: attention weights → "which stocks drove this prediction".

**You learn:** why most published stock-prediction results evaporate under costs.

## Phase 6 — The application
**Goal:** the thing you actually link in your job application.

- **Backend:** FastAPI. Endpoints: `/universe`, `/graph?date=`, `/predictions?date=`,
  `/backtest`, `/explain?ticker=&date=`. Load the trained checkpoint at startup; a nightly job
  refreshes data and re-runs inference.
- **Frontend:** React + Vite + TypeScript, with:
  - a force-directed / WebGL network view (Cytoscape.js or sigma.js) — **node color = predicted
    direction, node size = conviction, edge weight = learned adjacency**, with a date slider that
    animates the graph through time. This is the visual that makes people stop scrolling.
  - equity curve + drawdown chart, metrics panel, per-stock drilldown with attention explanation.
- Dockerize; deploy the demo (Fly.io / Render free tier) so the link works without setup.

**Intermediate step:** a Streamlit version in Phase 5 so you can look at results early — then
rebuild properly in React. Don't let the pretty UI block the modeling.

---

## Proposed repo layout

```
stocks_prediction/
├── marketgraph/
│   ├── data/          # ingest.py, universe.py, cache.py
│   ├── features/      # node_features.py, labels.py, splits.py
│   ├── graphs/        # correlation.py, sector.py, leadlag.py, learned.py
│   ├── models/        # gcn.py, gat.py, stgnn.py, baselines.py
│   ├── train/         # loop.py, losses.py, walkforward.py
│   ├── eval/          # metrics.py, backtest.py
│   └── api/           # FastAPI app
├── frontend/          # React app (Phase 6)
├── notebooks/         # one per phase — the learning log
├── configs/           # YAML experiment configs
├── tests/             # leakage tests, shape tests, graph tests
└── ROADMAP.md
```

## Stack decisions

- **PyTorch only, no PyG.** With ~100 nodes the graph is dense; sparse-scatter machinery buys
  nothing and hides the math. Add PyG later only if you scale to 1000+ tickers.
- **Parquet + pandas/polars** for data. No database until it hurts.
- **Hydra or plain YAML** configs + MLflow (or a CSV of runs) for experiment tracking.
- Everything CPU-trainable on your Mac; `mps` device for the temporal models.

## Suggested pace

| Weeks | Phases | Deliverable |
|---|---|---|
| 1 | 0–1 | Clean data + splits + leakage tests |
| 2 | 2 | Baseline table in the README |
| 3–4 | 3–4 | GCN → GAT → temporal GNN, ablated against baselines |
| 5 | 5 | Walk-forward backtest, seed variance, plots |
| 6–7 | 6 | FastAPI + React app, deployed |
| 8 | — | README, blog post, demo video |
