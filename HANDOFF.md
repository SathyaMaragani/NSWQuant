# NSEQuant — handoff for the next AI agent

Written 2026-11-14 ~02:30 IST. Read this whole file before touching anything. Everything here was checked against the
repo, the saved results and the running processes at that time; anything time-sensitive is marked **(as of writing)**.

---

## 1. What this project is, in one paragraph

A research project (built for a review panel) on **short-horizon order-book prediction and market making**. It has:
a tiny integer-only ML model (532 bytes, 18→8→8→3 int8 MLP) that predicts down/flat/up over the next 10 book
events; a C port of it (0.12 µs per tick); an HFT exchange simulator (`hftsim.py`) with queue position, latency,
fees and three infrastructure profiles; six strategies (A, C, Q, O, X, T) compared on recorded Bybit data; a
nine-day out-of-sample test; a live **paper-trading** dashboard on Bybit's public ETHUSDT feed (simulated orders,
nothing is ever sent to an exchange); a Results page; a NIFTY futures live-prediction dashboard; and a 5-slide deck.
The headline result is **algorithm X** (an order-book-imbalance market maker driven by Binance's ±2.5 % depth
imbalance): **+727.03 USDT over 9 out-of-sample days on ETH, 6/9 days positive, but only with a −0.5 bp maker rebate**.

---

## 2. Hard rules from the user (non-negotiable)

1. **Never place real orders and never trade real money.** Collectors, backtests and *simulated* paper trading only.
   (The user once wrote "do perform any real market trades with real money"; it was clarified as **do NOT**.)
2. **Never write API keys, secrets, tokens or proxy credentials into any file, log or field.** Keys come only from
   the user's own environment variables (`KITE_API_KEY`, `KITE_API_SECRET`, …). `.kite_token`, `.upstox_token`,
   `.env`, `data/`, `raw/`, `fi2010/` are git-ignored and must stay out of git.
3. **"Don't make it fake."** Every number shown must come from a real run. Profit that depends on the maker rebate
   must be labelled as such, with the retail-fee number next to it. No tuning on the test days, no hiding losing days.
4. **"Don't work on NIFTY for now"** (given earlier; the user later only asked to *start* the NIFTY dashboard). Don't
   develop the NIFTY side unless asked.
5. When the user says **stop**, stop the work but **keep running processes, browser tabs and open (simulated) orders
   alive** — they said exactly this: "keep the tabs and orders placed open and stop it".
6. Don't commit or push unless asked. (Everything up to this handoff is committed and pushed — see §4.)

---

## 3. The user and how they work

- Presents this to a review panel; wants things **demo-ready, impressive, and honest**. Asks short, casual, sometimes
  typo-heavy questions ("Alogirthm", "tradr"); infer intent, don't nitpick.
- Frequently asks for plain-language explanations of the algorithm to answer panel questions (see §9 glossary and
  the "traps" in §8).
- Has asked for: light, professional "quant/HFT firm" UI; Bookmap-style heatmap; a trading-platform view (order book,
  trade book, positions); a simple animated loader (replaced an elaborate canvas one).
- Pending decisions the user has **not** answered yet: connect a broker demo account (§11)? continue the
  retail-strategy research (§11)?

---

## 4. Repository and git state

- Repo: `D:\5Projects\NSEQuant` (Windows 11; shell tools: PowerShell 5.1 and Git Bash; Python 3.10.9 at
  `C:\Users\lucky\AppData\Local\Programs\Python\Python310\python.exe`). Remote: `origin` →
  `github.com/SathyaMaragani/NSWQuant` (note the repo name spelling).
- Branches: **`backend`** (current, main working branch; code + web pages) and **`frontend`** (web pages + sample JSON
  data contracts only: `web/*.html`, `web/fx.*`, `web/paper_state.json`, `web/paper_heat.json`, `web/state.json`,
  `web/results_data.json`, vendored ECharts/GSAP). Both are **pushed to `origin`**: backend at `c946934` plus small
  HANDOFF-only commits after it, frontend at `664c821`.
- **All work is committed on `backend`** in `c946934` ("Live algorithm X with a live Binance signal, trading desk,
  retail-fee strategy research, handoff doc"). Its page changes are **mirrored to `frontend` in `664c821`**, together
  with a fresh `paper_state.json` / `paper_heat.json` snapshot (with the new `desk`, `strategies.X`, `x` fields) and the
  frontend README's data contract. `web/dashboard.html` differs between the branches but was not part of this work and
  was left alone. What `c946934` contains:

| file | change | made by |
|---|---|---|
| `paper_trade.py` | Blotter, TrackedMM, TrackedExchange, BinanceSignal, LiveX; algorithm X live; trading-desk payload; selftests | this agent |
| `web/trading.html` | trading desk replaces the hero depth graph; X as headline; "Algorithm X + Model" label; chart follows desk selection; zoom-button scoping fix; loader status lines | this agent |
| `web/fx.js`, `web/fx.css` | old canvas "burn" loader removed, simple loader added | this agent |
| `strategies.py` | new `SignalSwing` (strategy D) | this agent |
| `retail.py` (new), `results/retail/` (new) | retail-fee strategy research runner and round-1 results | this agent |
| `README.md` | "Live paper trading" section; file table updated | this agent |
| `web/results.html` | hero now shows "the Binance signal · profit it added" (X − X0) and the KPI shows X's net | **not this agent** (another session or the user); committed as part of the working tree at the user's request — don't revert it, ask if unsure |
| `HANDOFF.md` | this file | this agent |

- Future commits: commit to `backend`; the user's earlier pattern was one descriptive commit per feature. Page changes
  were previously mirrored to the `frontend` branch (`web/trading.html`, `web/fx.js`, `web/fx.css`, and sample data
  if the state schema changed, plus the frontend README's data contract) — keep doing that, using a separate
  `git worktree` so the shared checkout never switches branch. Don't push unless asked.
  End commit messages without attribution lines.

---

## 5. What is running right now (as of writing)

| process | command | port / where | notes |
|---|---|---|---|
| Paper trader | `python -u paper_trade.py --port 8770` | http://127.0.0.1:8770/ (Results page at `/results.html`) | runs in the user's Terminal-panel tab titled **"paper trader"** (tab id `c19`). Started ~00:46 IST. |
| NIFTY dashboard | `python -u live_predict.py live --port 8765 --no-keepwarm` | http://127.0.0.1:8765/ | not started by this agent; replays the day's recording at 20× when NSE is closed |

**Live snapshot at 02:28 IST, uptime 1 h 42 m** (session P&L, simulated, 0.1 ETH orders):

| strategy | before fees | net | notes |
|---|---|---|---|
| X (Algorithm X + Model) | +7.92 | **+12.54** at the MM fee tier (−0.5 bp rebate, as tested) | **−10.57 at retail fees**; 556 fills; position +1.4 ETH; Binance z = +1.38 |
| C NSQ-MM | −21.82 | −62.76 at retail fees | |
| A no model | −37.77 | −115.41 at retail fees | |
| T taker | +5.34 | −147.15 at retail fees | |

Important operational facts:
- **Restarting the paper trader resets every session P&L, position and order to zero, and X needs ~17 min to start
  trading again** (15 min Binance-book warm-up + 4 snapshots × 30 s). The user wants it kept running — avoid restarts;
  if a server-side change is unavoidable, ask first.
- `serve()` copies `web/trading.html → results/paper/index.html` and `results.html`, `results_data.json`, `fx.js`,
  `fx.css`, `vendor/` into `results/paper/` **only at startup**. To ship a page-only change without restarting:
  `cp web/trading.html results/paper/index.html` (and `cp web/fx.js web/fx.css results/paper/`).
- The page polls `state.json` every 0.7 s; the engine writes it every 0.5 s and `heat.json` every 5 s (thread).
- To stop it: Ctrl-C in its terminal tab (the websocket loop may need the tab closed).
- The in-app Browser pane was **hidden** at writing; a hidden pane does not play CSS animations or `requestAnimationFrame`,
  so screenshots look frozen. Verify via computed styles, or ask the user to show the pane.

---

## 6. Architecture and files

```
Bybit public archive ──fetch_bybit.py──► data/markets/bybit_<cat>_<SYM>_<day>.npz  (top-5 book events, R(n,23), ts, tick)
Bybit public trades ─────────(curl)────► raw/trades/<SYM>_<cat>_<day>.csv.gz
Binance bookDepth archive ───(curl)────► raw/binance/bookDepth/<SYM>-bookDepth-<day>.zip  (±0.2..5 % depth, every ~30 s)
                                   │
tinyml.py (features, labels, int8 MLP)   strategy.py (MARKETS, FEES, model_probs, signals, prep, load_trades, MarketMaker)
                                   │
hftsim.py (Exchange simulator, profiles, replay, report, book_imbalance_z, binance_imbalance_z) + strategies.py
        ├─ hftsim.py main ─► results/hftsim/summary.csv, tuning.csv  (tune on 9 Nov tail, test once on 10 Nov)
        ├─ multiday.py ────► results/hftsim/multiday.csv, multiday_curves.json, x_delay.csv  (frozen, 9 days)
        └─ retail.py ──────► results/retail/*  (strategy D research, unfinished)
make_results.py ─► web/results_data.json ─► web/results.html (Results page)
paper_trade.py (live, Bybit public WS + Binance public depth) ─► results/paper/state.json, heat.json ─► web/trading.html
live_predict.py (NIFTY, reads collect_kite.py CSVs) ─► results/live/ ─► web/dashboard.html
```

| file | role |
|---|---|
| `tinyml.py` | streaming features (18), future-only labels, integer-only int8 MLP (training, quantisation, numba kernels), C export |
| `strategy.py` | `MARKETS` (ETH linear 0.1 unit; BTC spot 0.002 unit), `FEES` (VIP0: ETH 2.0/5.5 bp, BTC 10/10), `model_probs`, `signals`, `prep`, `load_trades`, the older `MarketMaker` used by live A/C |
| `hftsim.py` | the exchange simulator + experiment driver (`python hftsim.py`) |
| `strategies.py` | `QuotingMM` (A), `NSQMM` (C), `QueueAwareMM` (Q), `ImbalanceMM` (O and X), `SignalSwing` (D, new), `SignalTaker` (T) |
| `multiday.py` | 9-day frozen test; `CHOSEN` dict holds X and Q settings |
| `make_results.py` | builds `web/results_data.json` |
| `paper_trade.py` | live paper trading engine + HTTP server for the dashboard |
| `retail.py` | strategy D tuning/holdout runner (new, unfinished) |
| `live_predict.py`, `collect_kite.py`, `collect_upstox.py`, `nse_session.py` | NIFTY side (frozen per rule 4) |
| `fetch_bybit.py` | Bybit order-book archive → npz (resumable `.part` downloads) |
| `experiments.py`, `markets.py`, `tinyml_midprice.ipynb`, `c/` | model research, multi-market loaders, notebook, C kernel + bench |
| `web/` | `trading.html` (paper trading), `results.html`, `dashboard.html` (NIFTY), `fx.js`/`fx.css` (shared effects), `vendor/` (ECharts, GSAP) |

---

## 7. The algorithms

All prices inside the simulator are integer ticks; X/O settings are in basis points of the mid (1 bp = 0.01 %).

| id | class (file) | what it does |
|---|---|---|
| **A** | `QuotingMM` (strategies.py) | quotes both sides around mid − inventory skew; no model; the baseline |
| **C** NSQ-MM | `NSQMM` | A + TinyML: fair += κ·E[move]; pulls the side about to be run over on a confident call |
| **Q** NSQ-MM-Q | `QueueAwareMM` | C that only rests near the front of the queue (`max_ahead`), vetoes a side if P(adverse) > τ, needs `imb_min` of touch depth |
| **O** | `ImbalanceMM` | port of hftbacktest's "Market Making with Alpha – Order Book Imbalance" (MIT) using Bybit's own 200-level book; its tuned setting turned the signal off (c1 = 0) |
| **X** | `ImbalanceMM` with `zcol=2` | same code fed by **Binance ±2.5 % depth imbalance z** (mean of the 2 % and 3 % bands, z over 1 h, usable 1 s after each ~30 s snapshot) |
| **X0** | `ImbalanceMM`, `c1_bp=0` | X with the signal off — used to measure what the signal is worth |
| **T** | `SignalTaker` | crosses the spread on the model's most confident calls, holds 10–30 events |
| **D** (new) | `SignalSwing` | slow directional trade on the Binance z for a retail account (see §11) |

**X in detail** (`ImbalanceMM.targets`, called once per second):
```
tpb   = mid * 1e-4                                   # ticks per bp
alpha = aux[zcol]  (NaN → 0 in backtests)
res   = mid + c1_bp*tpb*alpha − skew_bp*tpb*pos/size
bid   = min(floor(res − half_bp*tpb), best_bid)      # post-only, never better than the touch
ask   = max(ceil (res + half_bp*tpb), best_ask)
a side stops quoting at ±max_pos lots (20 lots = 2 ETH at 0.1 ETH)
```
Frozen settings (`multiday.CHOSEN`, chosen on the 9 Nov tail, hft_firm profile):
ETH X `half_bp=1.0, c1_bp=2.0, skew_bp=0.02, zcol=2`; BTC X `half_bp=0.5, c1_bp=5.0, skew_bp=0.02, zcol=3`.
ETH Q `half=0.5, inv_skew=0.6, kappa=0.0, tau=0.25, imb_min=0.5, max_ahead=0.5, guard=True`.
Intuition: with ETH's 1-tick spread and c1 > half, whenever |z| > 0.5 the favoured side sits at the touch and the other
stands far back — X behaves like a **slow directional strategy trading passively**, often hitting the ±2 ETH cap.

**TinyML model** (C, Q, T only — **X does not use it**): 18 features (spread, microprice, imbalance L1/L3/L5, OFI 1/5/20,
returns 1/5/20, realised vol, volume delta, tbq/tsq imbalance, events since move, deep OFI, signed volume); label =
mean mid over the next 10 events vs now, flat band = bottom third of |moves|; trained on the first 85 % of 9 Nov.

---

## 8. Simulator and results

**hftsim profiles** (`hftsim.profiles`): retail = 50 ms latency, back of queue, VIP0 fees, 10 msgs/s, book-update
decisions only; **hft_firm** = 10 µs, front of queue if the level is < 100 ms old ("race"), maker **−0.5 bp** / taker
2 bp, 2,000 msgs/s, reacts to trades; frictionless = 0 latency, front, 0 fees. hft_firm numbers are **assumptions**.
Fill rules: join behind visible size; trades at our price consume queue first; traded-through or a snapshot crossing
our price = full fill; same-millisecond prints are one atomic sweep; shadow touch between snapshots.

**Test day 10 Nov, ETH, hft_firm, net USDT:** X +201.57 · Q +11.54 · T −10.26 · O −100.09 · C −232.10 · A −652.25.

**Nine out-of-sample days (3–8, 10–12 Nov), hft_firm, net USDT:**

| | ETH | BTC |
|---|---|---|
| X | **+727.03** (6/9 days, t = 1.83; gross −110.83, rebate +837.86) | −35.08 |
| X0 | −620.12 | −859.33 |
| Q | −95.39 (3/9) | +36.76 (4/9) |
| signal value X − X0 | **+1,347.15, 8/9 days, t = 3.57** | **+824.25, 8/9 days, t = 4.06** |

Note: the 9 days include the 10 Nov test day; excluding it ETH X is +525.47 over 8 days.

**Signal delay check (ETH, 10 Nov, net):** 1 s +201.57 · 30 s +212.17 · 60 s +193.46 · 5 min +4.92 · 1 h −280.69 →
fades over minutes (a timestamp leak would vanish at the first delay).

**Model:** ETH accuracy 58.5 % vs 50.3 % majority; 77.9 % on its most confident 10 %; C full tick 0.12 µs p50 / 0.20 µs
p99; 532 bytes.

**Traps when presenting (tell the user / never overclaim):**
- X's profit **needs the rebate**; at retail fees it loses. Rebates require a market-maker programme/high VIP tier.
- t = 1.83 over 9 days is **not significant** at 95 %. BTC has no edge.
- 58.5 % accuracy and 0.12 µs belong to the **TinyML model (C/Q/T), not X**. The user renamed X on the paper page to
  "Algorithm X + Model"; if asked "which model", it's the Binance imbalance z-score signal model.
- Everything is simulated against real recorded/live data; nothing touches an exchange.

---

## 9. Live paper trading (`paper_trade.py` + `web/trading.html`)

Engine (`Engine`), one process, Bybit public WS `orderbook.200` + `publicTrade` for ETHUSDT:
- **A and C**: `TrackedMM` = the backtest's `strategy.MarketMaker` (50 ms latency, VIP0 fees) with every order recorded in
  a `Blotter`. Selftest proves fills are identical to `MarketMaker`.
- **T**: `LiveTaker` (market orders, taker fee), also with a Blotter.
- **X**: `TrackedExchange` = `hftsim.Exchange` with the **hft_firm** profile (as tested) + `LiveX` (= `ImbalanceMM` with the
  frozen ETH settings, except it **sits out with no quotes when the signal is NaN** instead of quoting blind).
- **`BinanceSignal`**: rebuilds Binance USD-M ETHUSDT book (REST `fapi/v1/depth?limit=1000` snapshot + `@depth@500ms`
  diff stream, Binance's documented U/u/pu sync; gap → resync); every 30 s computes the ±2 %/±3 % band imbalance and a
  1 h z (ddof 1, ≥ 4 snapshots), usable 1 s later. REST 1,000 levels only reach ±0.6 %, so the far bands fill in from
  the diff stream: measured plateau after ~7–9 min at ±2 % ≈ 89–93k ETH, ±3 % ≈ 110–117k (Binance's archive shows
  ~92–95k / ~114–124k). Hence `X_WARM_S = 15*60`; history only starts after warm-up.
- `Blotter`: Kite-style order book (statuses PENDING/OPEN/COMPLETE/CANCELLED/REJECTED, modify keeps the order id),
  trade book (maker/taker, fee), average-cost positions (realised + unrealised = mark-to-mid equity, asserted).
- `state.json` top-level keys: `symbol, mode, venue, updated, uptime_s, book_events, trades, unit, tick,
  latency_ms_sim, fees_bps, params, last, series, heat, trades_heat, log, equity, strategies{A,C,T,X}, x, fills,
  latency, backtest, desk{depth, tape, books{C,A,T,X}{orders, trades, position, open, n_orders, n_trades}}`.
  `strategies.X` has `gross, fees (negative = rebate), net, fees_vip0, net_vip0, …, signal{z, imbalance, snapshots,
  synced, reach_pct, resyncs, warmup_s_left}`. Must stay NaN-free (`json.dumps(..., allow_nan=False)` is asserted).
- `series` rows are `[ts, mid, best_bid, best_ask, C_quote_bid, C_quote_ask]` (C only, by design).

Page (`web/trading.html`), top to bottom:
1. **Trading desk**: order book ladder (`renderBook` / `bookMaps` in trading.html: 10/20/50 levels, price grouping
   0.01/0.05/0.10/0.25, running totals per side, bid/ask imbalance meter, size-change flashes, walls = each side's 3
   largest levels ≥ 4× the median, spread in bp, last price with tick arrow; exact top 10 from `desk.depth`, deeper
   levels from the newest heatmap column, i.e. full depth only within ±150 ticks of the mid; our working orders tagged,
   our own price levels inserted as rows) · **Account** card: strategy selector (default **X**), X signal strip, an
   always-visible position line, and underline tabs **Positions** (table + detail: exposure, cap usage, break-even incl.
   fees, P&L split, activity) · **Open orders** · **Order history** (Executed / Cancelled filter) · **Trade book** (side and
   maker/taker filters, summary strip, position after each trade walked back from the live position, each trade marked
   to the current mid, CSV export) · **Time & sales** (min-size filter, buy/sell volume bar, ≥ 10 ETH prints highlighted).
   The strategy selector drives the books **and** the chart's working-order lines / fill markers. Chart order lines are
   accumulated in the page from the order book's OPEN orders on each poll (they rebuild after a reload, ≤ 2 min);
   fill markers come from the trade book.
2. KPI strip: X headline (net at MM tier + "at retail fees" in bold), C, A, T, adverse selection, decision time.
3. Depth heatmap (Bookmap-style, canvas bitmap inside ECharts) + "C · NSQ-MM, step by step" panel.
4. Algorithm terminal (X lines: SIGNAL on each Binance reading, FILL with rebate).
5. Session P&L chart, recent fills, backtest table.
- Loader: `FX.loader({status:[…]})` in `fx.js` — light, logo line draws itself, fades on first data (0.9–4 s, timer
  based so it closes in background tabs).
- Gotcha fixed: the heatmap zoom buttons used `.seg button` and overwrote every `.seg` handler; they are now scoped to
  `#zoomSeg`. Keep selectors scoped.

---

## 10. Other deliverables

- **Results page** `web/results.html` (served at `/results.html`): all backtests, nine-day table and curves, delay
  check, accuracy, latency, live-session view (`?state=` supported). Data from `make_results.py`.
- **NIFTY dashboard** `web/dashboard.html` via `live_predict.py` (port 8765).
- **Deck** (private claude.ai artifact, 5 slides: algorithms, usage, code, formulas, results, with speaker notes):
  https://claude.ai/artifact/Di4wWTZVLLMt7fcjjLVJn2 — numbers there match §8. It is private until the user shares it.

---

## 11. Open work — where things stopped

**A. Retail-fee strategy research (stopped by the user mid-run).** Goal stated by the user: "make it profitable for
base price for the normal trader API". Facts established:
- Market making at retail fees is structurally impossible on ETH: spread 1 tick (0.01 USDT ≈ 0.05 bp) vs 2 bp maker fee.
- Signal edge study (the 10 November development days, Binance z vs forward Bybit mid move, signed; script was a scratch file, not in the repo): |z|>1.5 (22 % of time): 5 min 1.9 bp,
  15 min 4.6, 30 min 7.9, 60 min 8.0; |z|>2: 30 min 9.9 bp. Round-trip costs: 4 bp maker/maker, 7.5 maker/taker, 11 taker/taker.
- Strategy D = `strategies.SignalSwing`; runner `retail.py` (retail profile, 1 ETH orders). Protocol: tune on the 10
  November days (`DEV`), freeze into `results/retail/chosen.json`, then **run once** on 14 never-used days
  (`HOLDOUT` = 20 Oct – 2 Nov) with `python retail.py holdout`.
- **Round 1** (36 settings, saved in `results/retail/tuning_round1.csv` and `tuning.csv`): best `k_in=2.0, hold_s=3600,
  entry_s=300, exit_s=600` → **+113.66 USDT over 10 days, only 4/10 days up, t = 0.46** (gross 395.50, fees 281.83).
  Not a reliable edge yet. `chosen.json` currently holds this setting.
- **Round 2** (16 settings, adds `extend=True` "hold while the signal still points our way" and `k_out`) is defined in
  `retail.GRID` but was **interrupted before writing results**. Running `python retail.py tune` now runs round 2 and
  **overwrites** `tuning.csv`, `tuning_days.csv` and `chosen.json` (round 1 copies are kept as `*_round1.csv`).
- Holdout data **(as of writing)**: trades + Binance files for all 14 days; Bybit book npz only for **20–26 Oct** (7 days).
  Finish with
  `python fetch_bybit.py linear ETHUSDT 2026-10-27 2026-10-28 2026-10-29 2026-10-30 2026-10-31 2026-11-01 2026-11-02`
  (resumable; existing days are skipped). That is the only data still missing.
- Honesty: report how many settings were tried on DEV (36 + 16), run the holdout **once**, and report whatever it shows.
  Never re-tune after looking at the holdout.

**B. Broker demo trading (proposed, awaiting the user).** Recommended **Bybit Demo Trading** (`https://api-demo.bybit.com`,
private WS `wss://stream-demo.bybit.com`; public data = mainnet; place/amend/cancel, executions, positions supported;
demo funds via `POST /v5/account/demo-apply-money`). The user must create the demo account and a demo API key
themselves (Contracts orders + positions, no withdrawals) and set `BYBIT_DEMO_KEY` / `BYBIT_DEMO_SECRET` in their own
shell. Plan: hard-code the demo domain, selftest that fails if a mainnet trading URL appears, post-only 0.1 ETH,
±2 ETH cap, show Bybit's own order ids/fills/position on the desk. **Tell the user: demo charges normal VIP0 fees, so X
is expected to lose there; it proves the pipeline, not profitability.** Zerodha has no paper API; Binance futures
testnet has a separate thin book.

**C. Git**: nothing outstanding — both branches are committed, mirrored and pushed (§4).

---

## 12. Verify before you claim anything

```bash
python hftsim.py --selftest
```
```bash
python paper_trade.py --selftest
```
```bash
python retail.py --selftest
```
(`paper_trade.py --selftest` prints a harmless "binance snapshot failed: offline" from an intentional offline test.)
Other selftests exist in `tinyml.py`, `strategy.py`, `fetch_bybit.py`, `live_predict.py`, `collect_kite.py`,
`collect_upstox.py`. Full re-runs: `python hftsim.py` (≈ minutes per market), `python multiday.py && python make_results.py`.

---

## 13. Gotchas

- Windows paths; Git Bash `sed -i` works; PowerShell 5.1 has no `&&`. Long jobs: run in the background.
- `results/paper/` is git-ignored and is a *copy* of `web/` made at server start (see §5).
- `state.json` must never contain NaN (the browser's `JSON.parse` rejects it).
- The engine's `feed_p50_ms` (~2 s) is mostly local clock offset vs Bybit timestamps, not real lag.
- `hftsim.FAR = 1e12`: a queue_ahead ≥ 1e11 means "behind the visible book".
- Binance REST depth (`limit=1000`) spans only ~±0.6 % of ETH's price; the ±2–3 % bands need the diff stream.
- Restarts reset the live session and X's warm-up (§5). Ask before restarting.
- Don't revert the `web/results.html` headline change (made outside this agent, now committed) without asking (§4).

---

## 14. Glossary (for answering the user's questions)

- **bp** (basis point) = 0.01 %. ETH traded ≈ 1,950–2,080 USDT in this data → 1 bp ≈ 0.2 USDT ≈ 20 ticks of 0.01.
- **Maker / taker**: maker rests in the book (adds liquidity); taker trades immediately. **Rebate**: exchange pays the
  maker (negative fee), funded by taker fees — not a refund of our money.
- **Imbalance** = (bids − asks)/(bids + asks) in a price band; **z** = (imb − 1 h mean)/(1 h std).
- **Adverse selection / markout**: how far the mid moves against a fill after N seconds.
- **X0**: X with the signal off; **signal value** = X − X0.
- **Out of sample**: days never used for choosing settings.
