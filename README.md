# NSEQuant — TinyML order-book mid-price prediction and HFT market making

Integer-only int8 model (532 B) that predicts the next mid-price move from a 5-level limit order book,
~0.14 µs per tick in C. Evaluated on NIFTY futures (live), FI-2010, Bybit BTC spot and ETH perpetual.
On top of it, **NSQ-MM**: a prediction-driven, inventory-aware market maker, backtested on tick data and
paper-traded live on Bybit's public feed. Nothing here ever sends an order.

## Branches
- `backend` (protected) — everything: collectors, model core, C kernel, notebook, experiments, live engines.
- `frontend` — only the dashboards (`web/`), with sample state files so they run without the backend.
  Backend picks up UI changes with `git checkout frontend -- web`.

## The trading algorithm (NSQ-MM)
On every order-book update:
1. **Predict** — the TinyML model gives p(down), p(flat), p(up) for the next 10 book events.
2. **Price** — `fair = mid + κ·E[move] − skew·position` (Avellaneda–Stoikov inventory control plus our forecast).
3. **Quote** — post-only bid `floor(fair − half)` and ask `ceil(fair + half)`; keep queue priority when unchanged;
   **pull** the side the model confidently says will be run over (adverse-selection protection).
4. **Protect** — position cap, order-rate cap, latency budget.

Also tested: **T**, a signal taker that crosses the spread only on the model's most confident calls.
Backtest (`strategy.py`): event-driven replay of book + trade ticks, queue position at each price level,
50 ms order latency, fees per venue, post-fill markouts. Parameters tuned on 9 Nov 2026, tested once on 10 Nov.

| Test day, P&L before fees (USDT) | ETH perpetual | BTC spot |
|---|---|---|
| A  market maker, no model | −1,184.60 | −560.54 |
| C  NSQ-MM | −743.81 (−37% loss) | −329.03 (−41% loss) |
| T  signal taker | +65.43 | +44.32 |
| T  at 200 ms / 1 s latency | +37.42 / −0.79 | +34.38 / +16.32 |

At Bybit VIP0 retail fees every variant is negative; the model's value is fewer toxic fills (market maker)
and a gross edge that decays with latency (taker). Full table: `results/trading/backtest.csv`.

## HFT simulation environment (`hftsim.py` + `strategies.py`)
Replays the real book snapshots and every trade print, so strategy work is separated from infrastructure.
Infrastructure is a switch:

| profile | order/cancel latency | queue position | fees (maker / taker) | reacts to |
|---|---|---|---|---|
| `retail` | 50 ms | back of the queue | Bybit VIP0 | book updates |
| `hft_firm` | 10 µs (co-location) | first at a level that just formed | −0.5 bp rebate / 2 bp | every trade |
| `frictionless` | 0 | front of every queue | 0 / 0 | every trade |

`hft_firm` values are assumptions standing in for co-location and a market-maker fee tier, not measurements.
Write a strategy by subclassing `hftsim.Strategy` (`on_book`, `on_trade`, `on_fill`) and using
`env.place / replace / cancel / market`. Each strategy is tuned per profile on 9 Nov and tested once on 10 Nov.
The tuning grids are fixed in `hftsim.run_market`.

Net P&L after fees on the test day (USDT):

| | ETH retail | ETH HFT firm | ETH frictionless | BTC retail | BTC HFT firm | BTC frictionless |
|---|---|---|---|---|---|---|
| A market maker, no model | −3,586.74 | −652.25 | −1,361.52 | −5,413.84 | −343.00 | −619.95 |
| C NSQ-MM | −2,096.43 | −232.10 | −411.45 | −3,359.08 | −146.12 | −293.14 |
| Q NSQ-MM-Q (queue-aware) | −107.71 | +11.54 | +65.73 | −24.38 | −0.48 | −8.88 |
| O imbalance MM, hftbacktest port (Bybit book) | −1,131.11 | −100.09 | −302.01 | −1,552.63 | −83.30 | −195.98 |
| **X imbalance MM, Binance signal** | −151.06 | **+201.57** | +175.68 | −524.54 | **+2.15** | −16.63 |
| T signal taker | −27.00 | −10.26 | +15.49 | −29.70 | −37.08 | +71.29 |

Infrastructure cuts NSQ-MM's net loss by about 90% (retail → HFT firm). The model cuts the loss by 42–65% under every
profile. Full output: `results/hftsim/`.

### NSQ-MM-Q: the market maker that is profitable under `hft_firm` (ETH)
Designed from a per-fill diagnosis on day-1 data only (7k fills, each with what was known before it):
- fills from the back of a queue only happen when a sweep runs through the level: −20 ticks per ETH fill at 10 s;
- among front-of-queue fills, the quarter the model rated least likely to be run over earned +22 ticks
  (rebate included); the quarter it rated most likely lost −13 (ETH) / −47 (BTC).

So, per side: rest only near the front of the queue (cancel if more than half a lot is ahead), quote only if the
model's probability of a move against that side is ≤ τ, and at the touch only if our side holds enough of the depth.
Re-quote on every trade. One grid (48 settings), the same for both markets, tuned on day 1 and tested once on day 2.

What it shows, honestly:
- **ETH: +11.54 net on the test day** (4,449 fills); BTC −0.48, i.e. break-even, not profitable.
- The profit is the maker rebate: gross is −30.43 (ETH), and with no rebate the net is −30.43.
  What the model buys is selection: adverse markout drops from about −20 to −17.5 ticks per fill, and fills drop
  from 75k to 4k, so the rebate (10.7 ticks per ETH fill) now covers the remaining adverse selection.
- It does not depend on the optimistic queue rule: with a 10 ms race window +4.45, at the back of every queue
  +7.50, with 1 ms latency +9.58 (ETH).
- It is thin: the test-day max drawdown (−52.74) is larger than the profit. One test day is not evidence of a
  durable edge; it needs more days before anyone relies on it.

### O and X: an open-source strategy, ported and tested
O is the order-book-imbalance market maker from hftbacktest's tutorials (github.com/nkaz001/hftbacktest, MIT), which
reports Sharpe 3–11 on Binance Futures backtests with a maker rebate:
`alpha = z-score of (bid depth − ask depth)`, `fair = mid + c1·alpha`, `reservation = fair − skew·position`,
quote `reservation ± half_spread` (never better than the touch), requote once a second.
- **O**, the port as published but fed by Bybit's own book (all 200 published levels, ~0.1 % of the price): the
  signal barely predicts Bybit's next moves (correlation ≈ 0.005 on day 1), tuning switches it off, and the test
  day loses (ETH −100.09, BTC −83.30).
- **X**, the same strategy fed by the signal the tutorial actually uses: imbalance within ±2.5 % of the price, from
  Binance's public `bookDepth` archive (30 s snapshots, data.binance.vision; files in `raw/binance/`, not committed).
  On day 1 it predicts Bybit's next 5 minutes (correlation +0.13 ETH, +0.08 BTC). Test day, `hft_firm`:
  ETH +201.57 net (+90.40 before fees), BTC +2.15. 10 Nov turned out to be one of X's better days; see below.

Checks on X (ETH, test day, `hft_firm`):
| check | net P&L |
|---|---|
| as tested (signal usable 1 s after each Binance snapshot) | +201.57 |
| Binance signal 30 s / 60 s late | +212.17 / +193.46 |
| 5 min late / 1 hour late | +4.92 / −280.69 |
| no maker rebate | +90.40 |
| 10 ms queue race / back of every queue / 1 ms latency | +201.35 / +200.83 / +201.57 |

The profit survives a 60 s delay and is gone at 5 min, which is what a real minutes-ahead signal looks like (a
timestamp leak would vanish at 30 s); a stale signal loses. Caveats: it earns by holding inventory in the signal's
direction for minutes (10 s markout is still −17 ticks per fill), so it is part market making, part slow directional
bet; the test-day max drawdown (−191.10) is about as large as the profit; it is one test day, and several strategy
families (A, C, Q, O, X) have now been tried, so the best of them is flattered by selection. It needs more days.

### Nine out-of-sample days (`multiday.py`)
The 9 Nov settings, frozen, replayed under `hft_firm` on 3–8 and 10–12 Nov (no re-tuning). Net USDT:

| | ETH | BTC |
|---|---|---|
| X, net (days positive, t) | **+727.03** (6/9, t = 1.83) | −35.08 (5/9, t = −0.26) |
| X, before fees and rebate | −110.83 | −384.02 |
| X0, same quotes with the signal off | −620.12 | −859.33 |
| **signal value, X − X0** | **+1,347.15, better on 8/9 days (t = 3.57)** | **+824.25, better on 8/9 days (t = 4.06)** |
| Q, NSQ-MM-Q | −95.39 (3/9) | +36.76 (4/9) |

What holds: the Binance ±2.5 % signal is real; it beats the same quoting without it on 8 of 9 days in both markets.
What does not: across the 9 days X needs the maker rebate (ETH is −110.83 before it), ETH's t of 1.83 is not
significant at 95 %, BTC has no edge, and Q's +11.54 test-day result did not hold (3/9 days positive on ETH).
Per-day results: `results/hftsim/multiday.csv`; the Results page (`web/results.html`, served by `paper_trade.py`
at `/results.html`) shows all of it, built by `make_results.py`.

### Live paper trading
`paper_trade.py` runs X live next to A, C and T. X uses the same `hftsim.Exchange`, the same `ImbalanceMM` class and the
same frozen 9-Nov settings as the 9-day test (HFT-firm profile, −0.5 bp maker rebate). Its signal is rebuilt live:
Binance's USD-M book from a REST snapshot plus the diff stream, then every 30 s the ±2 % / ±3 % band imbalance,
z-scored over the last hour. That matches what the backtest read from the bookDepth archive.
Two live-only differences:
- X waits 15 min while the rebuilt book's far levels fill in, then needs 4 snapshots before it quotes.
- X sits out with no quotes whenever the signal is unavailable; the backtest quoted as if the signal were 0.

The page shows X's net at the rebate tier and, next to it, the same trades at retail VIP0 fees. Over the 9 days, X
was profitable only with the rebate, and 3 of 9 days lost money, so a single live session can be negative.
The top of the page is a trading desk: market depth with our working orders, positions, order book (every order and
its status), trade book (every execution) and time & sales.

Fix in this version: trade prints that share a millisecond are one sweep, so a cancel can no longer take effect
between them (the earlier simulator let the queue-depletion guard dodge part of a sweep). A and C were re-run.

## Main files
| file | what |
|---|---|
| `collect_kite.py` / `collect_upstox.py` | read-only live NIFTY order-book recorders (keys from env vars) |
| `fetch_bybit.py` | crypto order-book history (Bybit public archive) |
| `tinyml.py` | features, labels, integer-only int8 model, reliability tools, C export |
| `markets.py` | market loaders |
| `tinyml_midprice.ipynb` | full prediction pipeline and results |
| `experiments.py` | teacher vs student, QAT, Pareto, importance, decision value |
| `strategy.py` | NSQ-MM market maker, signal taker, event-driven backtester |
| `paper_trade.py` | live paper trading on Bybit's public ETHUSDT feed (A, C, T, and X with a live Binance signal) + trading-desk dashboard |
| `hftsim.py`, `strategies.py` | HFT simulation environment with infrastructure profiles; strategy library (A, C, Q, O, X, T) |
| `multiday.py`, `make_results.py` | frozen-settings test across 9 days; builds `web/results_data.json` for the Results page |
| `live_predict.py` | live / replay NIFTY inference + dashboard server |
| `c/` | C feature port, generated model header, benchmark |
| `web/` | dashboards (ECharts + GSAP, vendored) |

## Run
```
pip install numpy pandas numba torch scikit-learn scipy matplotlib kiteconnect websocket-client ziglang
python fetch_bybit.py linear ETHUSDT 2026-11-09 2026-11-10   # order books
# trade ticks: https://public.bybit.com/trading/ETHUSDT/  -> raw/trades/ETHUSDT_linear_<date>.csv.gz
python strategy.py                      # backtest, saves models/ethusdt_mm.npz
python paper_trade.py                   # live paper trading at http://127.0.0.1:8770/
python multiday.py && python make_results.py   # multi-day test -> Results page at http://127.0.0.1:8770/results.html

python collect_kite.py --login          # NIFTY: daily, keys in KITE_API_KEY / KITE_API_SECRET
python collect_kite.py                  # record during market hours
python live_predict.py train
python live_predict.py live             # dashboard at http://127.0.0.1:8765/
```
Data folders (`data/`, `fi2010/`, `raw/`) are not in the repo; recreate them with the scripts above.
