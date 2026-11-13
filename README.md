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
| **Q NSQ-MM-Q (queue-aware)** | −107.71 | **+11.54** | +65.73 | −24.38 | **−0.48** | −8.88 |
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
| `paper_trade.py` | live paper trading on Bybit's public ETHUSDT feed + dashboard |
| `hftsim.py`, `strategies.py` | HFT simulation environment with infrastructure profiles; strategy library |
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

python collect_kite.py --login          # NIFTY: daily, keys in KITE_API_KEY / KITE_API_SECRET
python collect_kite.py                  # record during market hours
python live_predict.py train
python live_predict.py live             # dashboard at http://127.0.0.1:8765/
```
Data folders (`data/`, `fi2010/`, `raw/`) are not in the repo; recreate them with the scripts above.
