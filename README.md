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
