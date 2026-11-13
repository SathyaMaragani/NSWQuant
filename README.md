# NSEQuant — dashboard (frontend branch)

Only the UI lives on this branch. Backend code is on the `backend` branch and is never touched from here.

## Run without the backend
```
python -m http.server 8000 -d web
```
Open http://127.0.0.1:8000/dashboard.html — it reads `web/state.json` (a snapshot from a NIFTY replay) once a second.
Edit `state.json` by hand to test edge cases (empty book, gaps, no probabilities).

Paper-trading page: http://127.0.0.1:8000/trading.html?state=paper_state.json (a live snapshot of the NSQ-MM paper trader).
Its data comes from `paper_trade.py` on the backend: `strategies {A, C, T}` (gross, net_vip0, fills, position, markout10s_ticks),
`last` (probs, mid, fair, alpha_term, inv_term, quote_bid, quote_ask, pull flags, book), `series` [ts, mid, bid, ask, our bid, our ask],
`equity {A, C, T}` [ts, P&L], `fills` [{t, strategy, side, price, qty}], `latency`, `params`, `backtest` rows.

## Data contract (`state.json`, published by `live_predict.py`)
`mode`, `symbol`, `updated`, `ticks`, `resolved`, `horizon`, `accuracy`, `rolling`, `majority`,
`last {t, mid, conf, pred, probs[down, flat, up]}`, `confident {n, accuracy}`, `confusion` (3x3, rows = actual),
`latency {compute_p50_us, compute_p99_us, lag_p50_ms, lag_p99_ms}`, `series[] {t, mid, bid, ask, pred, conf, out}`,
`book {bp, bq, ap, aq, spread, imb1, depth, tick}`, `hist {acc, lat, lag, conf}`,
`model {trained_ticks, val_accuracy, val_majority, val_persistence, arch, bytes, runtime}`.
Classes: 0 = down, 1 = flat, 2 = up. Keep these keys working; ask before needing new ones (they come from the backend).

Libraries are vendored in `web/vendor/` (ECharts 5.4.3, GSAP 3.12.5) so the page works offline.
