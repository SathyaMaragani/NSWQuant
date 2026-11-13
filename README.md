# NSEQuant — TinyML order-book mid-price prediction

Integer-only int8 model (532 B) that predicts the next mid-price move from a 5-level limit order book,
~0.14 µs per tick in C. Evaluated on NIFTY futures (live), FI-2010, Bybit BTC spot and ETH perpetual.

## Branches
- `backend` — everything: collectors, model core, C kernel, notebook, experiments, live engine.
- `frontend` — only the dashboard (`web/`). Work on UI there; it runs on its own with a sample `state.json`.
  Backend picks up UI changes with `git checkout frontend -- web`.

## Main files
| file | what |
|---|---|
| `collect_kite.py` / `collect_upstox.py` | read-only live NIFTY order-book recorders (keys from env vars) |
| `fetch_bybit.py` | crypto order-book history (Bybit public archive) |
| `tinyml.py` | features, labels, integer-only int8 model, reliability tools, C export |
| `markets.py` | market loaders |
| `tinyml_midprice.ipynb` | full pipeline and results |
| `experiments.py` | teacher vs student, QAT, Pareto, importance, decision value |
| `live_predict.py` | live / replay inference + dashboard server |
| `c/` | C feature port, generated model header, benchmark |
| `web/` | dashboard (ECharts + GSAP, vendored) |

## Run
```
pip install numpy pandas numba torch scikit-learn scipy matplotlib kiteconnect ziglang
python collect_kite.py --login          # daily, keys in KITE_API_KEY / KITE_API_SECRET
python collect_kite.py                  # record during market hours
python live_predict.py train
python live_predict.py live             # dashboard at http://127.0.0.1:8765/
```
Data folders (`data/`, `fi2010/`) are not in the repo; recreate them with the collectors / `fetch_bybit.py`.
Nothing here places orders.
