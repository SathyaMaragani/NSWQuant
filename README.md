# NSEQuant — dashboard (frontend branch)

Only the UI lives on this branch. Backend code is on the `backend` branch and is never touched from here.

## Run without the backend
```
python -m http.server 8000 -d web
```
Open http://127.0.0.1:8000/dashboard.html — it reads `web/state.json` (a snapshot from a NIFTY replay) once a second.
Edit `state.json` by hand to test edge cases (empty book, gaps, no probabilities).

Paper-trading page: http://127.0.0.1:8000/trading.html?state=paper_state.json&heat=paper_heat.json (a live snapshot of the paper
trader: algorithm X with the live Binance signal, plus A, C and T).
Results page: http://127.0.0.1:8000/results.html?state=paper_state.json — reads `results_data.json` (built on the backend by
`make_results.py`: strategies, per-day results and curves, delay check, accuracy, latency) and the live state for its last panel.
Its data comes from `paper_trade.py` on the backend: `strategies {A, C, T, X}` (gross, net_vip0, fills, position, markout10s_ticks;
X also has fees (negative = maker rebate), net at the MM fee tier, and `signal {z, imbalance, snapshots, synced, reach_pct,
resyncs, warmup_s_left}`), `x` (X's frozen settings: half_bp, c1_bp, skew_bp, zcol, profile, maker_bps, max_pos, unit),
`last` (probs, mid, fair, alpha_term, inv_term, quote_bid, quote_ask, pull flags, book), `series` [ts, mid, bid, ask, our bid, our ask],
`equity {A, C, T, X}` [ts, P&L], `fills` [{t, strategy, side, price, qty}], `latency`, `params`, `backtest` rows.
Trading desk: `desk {depth {bp, bq, ap, aq} (10 levels), tape [[ts, price, qty, aggressor side]], books {X, C, A, T}}`; each book is
`{orders [{id, s, t, tu, side, type LIMIT|MARKET, px, qty, filled, avg, status PENDING|OPEN|COMPLETE|CANCELLED|REJECTED, mods,
q (queue ahead; >= 1e11 = behind the visible book), note}], trades [{id, oid, s, t, side, px, qty, fee, maker, value}],
position {qty, avg, mark, unrealised, realised, fees, net, buy_qty, buy_avg, sell_qty, sell_avg}, open, n_orders, n_trades}`.
The chart's working-order lines and fill markers are drawn from `desk.books` for the strategy selected on the desk.
Order book ladder: exact top 10 from `desk.depth`, deeper levels from the newest `heat` column (full depth within ±150 ticks);
levels 10/20/50, price grouping, running totals, imbalance meter, size-change flashes, walls, spread in bp, last price from `desk.tape`.
Account card (all from `desk.books`): Positions (table + exposure, cap usage, break-even, P&L split), Open orders, Order history,
Trade book (side / maker-taker filters, summary, position after each trade walked back from `position.qty`, each trade vs the
current mid, CSV export). Time & sales (`desk.tape`): minimum-size filter, buy/sell volume bar, large prints.
Depth heatmap: `heat` = newest columns `[ts, lowest tick, best bid tick, best ask tick, sizes[301]]` (resting size at each tick
from the lowest one up, ±150 ticks around the mid, one column per 300 ms) and `trades_heat` = `[t0, t1, tick, buy volume, sell volume]`.
`state.json` carries only the newest columns; `heat.json` holds the last 2 minutes and is read once when the page opens.
Algorithm terminal: `log` = newest lines `{seq, t (ms), s: M model | C NSQ-MM | T taker | X algorithm X, k: SIGNAL | PLACE | AMEND | CANCEL | LIVE |
FILL | EXIT | ENTER | FLAT, m: text}`; `seq` only grows, so the page appends lines with a higher `seq` (a lower one means the engine restarted).

## Data contract (`state.json`, published by `live_predict.py`)
`mode`, `symbol`, `updated`, `ticks`, `resolved`, `horizon`, `accuracy`, `rolling`, `majority`,
`last {t, mid, conf, pred, probs[down, flat, up]}`, `confident {n, accuracy}`, `confusion` (3x3, rows = actual),
`latency {compute_p50_us, compute_p99_us, lag_p50_ms, lag_p99_ms}`, `series[] {t, mid, bid, ask, pred, conf, out, lv: [bp[5], bq[5], ap[5], aq[5]]}` (lv feeds the depth heatmap),
`book {bp, bq, ap, aq, spread, imb1, depth, tick}`, `hist {acc, lat, lag, conf}`,
`model {trained_ticks, val_accuracy, val_majority, val_persistence, arch, bytes, runtime}`.
Classes: 0 = down, 1 = flat, 2 = up. Keep these keys working; ask before needing new ones (they come from the backend).

Libraries are vendored in `web/vendor/` (ECharts 5.4.3, GSAP 3.12.5) so the page works offline.
`fx.js` / `fx.css` hold the shared effects for the paper-trading and Results pages: a short light loading screen (the logo's
chart line draws itself, status lines, a sliding bar; it fades out on the first data, 0.9-4 s), the live depth chart with fading history, the animated cumulative P&L lines, the ticker band, word reveals, cursor ring and card
spotlight. Reduced-motion users get the pages without the animation.

The paper-trading page now shows only algorithm X and the no-model baseline A (the backend still publishes C and T).
hftbacktest replay: `hbt.html` reads `x_replay.json`, `heat.bin` (float32, columns of 301 sizes) and `trades.bin` (float64 rows
[ms from window start, price, qty, side]) produced on the backend by `hbt/run_x.py`; no sample data is kept here (about 17 MB).
