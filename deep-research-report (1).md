# High-Frequency Trading Algorithms Used by Profitable Quantitative Firms

## Executive findings and evidence standard

The central finding is that the most profitable electronic-trading firms do **not** publicly disclose a single “secret HFT algorithm.” What is public, however, is enough to identify a fairly consistent production architecture: low-latency market making, short-horizon statistical prediction, cross-instrument relative-value trading, derivatives pricing and hedging, venue-aware execution, and portfolio-level risk control. At several firms, recent first-party disclosures now explicitly confirm the use of machine learning or neural networks for real-time price prediction rather than merely suggesting that such methods are plausible. citeturn634502search2turn956800search8turn911922search23turn448436search35

A useful distinction is therefore:

| Evidence category | Meaning in this report |
|---|---|
| **Confirmed** | The firm itself, a filing, regulator, or employee-authored publication explicitly says it performs the activity. |
| **Strong inference** | Multiple public facts strongly imply an implementation family, but the actual production model is undisclosed. |
| **Generic / academic** | A known quantitative method that fits the activity but **cannot** legitimately be attributed to a particular firm without evidence. |
| **Alleged / contested** | A regulator has described trading behavior, but the firm disputes the characterization or proceedings remain unresolved. |

The most striking recent disclosure is Jane Street's October 2026 description of its internal **Flex Model** system. Jane Street states that neural networks are used heavily in “many trading strategies” and that its feature system converts raw events—orders, cancellations and trades—into numerical features consumed by models; the post also says an early version was deployed in a live trading strategy in 2022. This is unusually direct evidence that event-stream neural-network prediction is actually used in production trading, although Jane Street does not reveal network architectures, labels or trading rules. citeturn634502search2

Hudson River Trading is similarly explicit: it says deep learning is core to its trading and describes real-time models operating on millions of irregular market events under ultra-low-latency constraints. HRT also publicly describes training models on years of exchange data and replaying them across large simulation workloads. citeturn956800search8turn956800search7 XTX Markets says that machine-learning models generate price forecasts for more than 53,000 instruments and that those forecasts support its algorithmic liquidity provision. citeturn911922search23turn911922search41 Jump describes its own core research problem as finding alpha in “high-volume, low-latency environments” with real-time inference over very large datasets. citeturn448436search35

Profitability evidence is also unusually strong for several firms. Reuters reported that Jane Street generated about **$39.6 billion of net trading revenue in 2025**; Bloomberg reported Citadel Securities generated **$12.2 billion of trading revenue and roughly $5.4 billion of profit** in 2025; Optiver itself reports **€4.556 billion net trading income and €1.769 billion net profit** for 2025. citeturn470137news49turn543849search2turn543849search4 Virtu reported **$2.145 billion of adjusted net trading income** in 2025, while Flow Traders reported **€485.8 million NTI and €133.6 million net profit**. citeturn925364search27turn925364search3 These results establish that the strategy families discussed below belong to highly profitable operations, but they do **not** prove that any particular publicly visible model is responsible for that profit.

## Firm-by-firm evidence matrix

The matrix deliberately avoids assigning an academic model such as Avellaneda–Stoikov to a firm merely because it is a market maker. Where a specific model family is not directly disclosed, the entry says so.

| Firm | Publicly supported strategy / algorithm family | Precise evidence | Confidence | Markets / instruments | Latency profile | Public profitability evidence |
|---|---|---|---|---|---|---|
| **Jane Street** | ETF and cross-asset market making; event-driven ML/neural-network features; quantitative valuation | Jane Street says it prices/trades >10,000 ETFs globally and uses quantitative models; 2026 Flex Model post states neural nets are used heavily in many strategies and market-event features feed models. citeturn956800search0turn634502search2 | **Confirmed** | ETFs, equities, bonds, options, commodities/crypto-linked ETPs | Some strategies clearly latency sensitive; Jane Street's exchange-engineering work discusses single-digit-µs components and high event rates, though that does not prove every strategy requires µs latency. citeturn956800search2 | ~$39.6bn 2025 net trading revenue reported by Reuters/Bloomberg. citeturn470137news49turn543849search0 |
| **Citadel Securities** | Large-scale equities/options market making; real-time predictive quantitative models; pricing, risk and execution systems | Firm describes statistical inference, ML and market-microstructure models driving real-time decisions; its options business uses sophisticated pricing models and sends >25bn option quotes/day. citeturn634502search3turn956800search3 | **Confirmed** | Equities, options, ETFs, fixed income, crypto | Options engineering roles explicitly state exchange colocation is needed because of latency requirements. citeturn956800search5 | $12.2bn 2025 trading revenue; ~$5.4bn profit reported by Bloomberg. citeturn543849search2turn543849search3 |
| **Hudson River Trading** | ML/deep-learning prediction; market making; factor/risk modeling; longer-horizon proprietary strategies also exist | HRT says deep learning is core to trading and operates under ultra-low-latency constraints; its employee-authored factor-model article shows portfolio-risk and expected-return modelling used internally, although not necessarily HFT alpha itself. citeturn956800search8turn634502search1 | **Confirmed for ML; partial for exact signals** | Multi-asset | Ultra-low latency for part of business; massive offline CPU/GPU research/simulation infrastructure. citeturn956800search7turn956800search8 | 2025 trading revenue reported around $12bn; private figures rather than audited public group accounts. citeturn470137search0turn470137news48 |
| **XTX Markets** | ML-driven price forecasting plus liquidity provision and systematic market making | XTX explicitly says statistical/ML models forecast prices across >53,000 instruments; client material says it can hold risk for meaningful periods rather than immediately hedge every trade. citeturn911922search23turn911922search41 | **Confirmed** | Equities, FX, fixed income, commodities, crypto | Tick-to-trade system is automated; exact latency undisclosed. citeturn911922search7 | UK Companies House accounts confirm profitable operating entities, although entity-level accounts should not be equated automatically with entire group economics. citeturn543849search5turn543849search10 |
| **Optiver** | Options/equities market making; ML pricing; volatility/risk hedging | Optiver explicitly describes market making as a pricing problem with hundreds of weak signals—trades, order-book changes, related instruments, competitors—combined by ML; it also publicly explains options volatility and hedging. citeturn634502search0turn911922search11turn911922search26 | **Confirmed** | Options, futures, equities, ETFs and other assets | Optiver's own ML article says competitors react within microseconds and execution realism must account for latency and queue position. citeturn634502search0 | €1.769bn net profit on €4.556bn NTI in 2025. citeturn543849search4 |
| **IMC** | HFT equity prediction, options market making, options/volatility valuation, ML signals | IMC jobs explicitly mention high-frequency low-latency equity strategies, predictive models used across options/equities, statistically robust options signals and a real-time valuation library on the critical path of HFT market-making systems. citeturn448436search1turn448436search0turn911922search14 | **Confirmed** | Equity options, equities, futures, commodities, ETFs, FX | Firm calls low latency a competitive edge; critical pricing libraries sit directly in HFT paths. citeturn448436search2turn911922search14 | Global financial details are less transparent; IMC India reported ₹298 crore PAT for CY2025 according to CRISIL. citeturn470137search4 |
| **Jump Trading** | High-frequency trading, statistical arbitrage, ML alpha, ultra-low-latency execution | Jump explicitly says its strategies range from high frequency through stat arb to discretionary macro; Jump AI says its core focus is alpha in high-volume low-latency settings and real-time inference. citeturn956800search6turn448436search35 | **Confirmed at family level** | Multi-asset | Very high: custom ASICs, FPGAs, optimized networking and colocated infrastructure are publicly described. citeturn448436search28turn448436search21 | Group profitability is not publicly disclosed in a clean consolidated form in the material reviewed. |
| **Virtu Financial** | Multi-asset electronic market making; ETF/options market making; execution services | SEC filings state market making across cash, futures and options in equities, FICC, crypto and commodities; annual reports describe high-speed direct connections to exchanges. citeturn911922search16turn911922search51 | **Confirmed** | Broad multi-asset | High-speed exchange connectivity is explicitly documented; exact strategy latency undisclosed. citeturn911922search51 | $2.145bn adjusted NTI in 2025; public listed-company reporting. citeturn925364search27 |
| **Flow Traders** | ETP/ETF market making and increasingly crypto/digital-asset liquidity provision | 2025 report says it remains a leading ETP liquidity provider and held >25% European ETP market share; earlier/current reports describe substantial crypto market making. citeturn448436search39turn448436search40 | **Confirmed** | ETFs/ETPs, crypto, increasingly broader assets | Technology-enabled electronic market making; exact µs profile not disclosed publicly. | €133.6m net profit and €485.8m NTI in 2025. citeturn925364search3 |
| **Tower Research** | Fully automated proprietary quantitative trading; market making; venue-specific FX pricing, last look and pre-hedging; multiple independent strategy teams | Tower states trading is almost exclusively fully automated; its FX disclosure explicitly describes last-look checks and pre-hedging; engineering page mentions ML, FPGA and dozens of colocation sites. citeturn709734search2turn709734search1 | **Confirmed** | Broad exchange markets; FX disclosure especially detailed | Strongly latency sensitive in parts of platform; C++ used for most latency-sensitive systems. citeturn709734search3 | Private; no comparable consolidated public profit figure found. |
| **DRW** | ETF market making, options/volatility market making, latency-sensitive proprietary trading, dispersion | DRW says it participates in latency-sensitive trading; it is an AP for >2,000 ETFs; current job material explicitly identifies systematic options liquidity and equity-index/single-stock dispersion trading. citeturn448436search16turn709734search7turn448436search4turn448436search13 | **Confirmed** | ETFs, futures, options, crypto, fixed income | Some business latency sensitive; ETF RFQ quoting can operate at 300–400ms, demonstrating not all profitable electronic trading is microsecond HFT. citeturn709734search7 | Private; UK subsidiaries do not represent global economics. |
| **Jane Street — India case** | Cross-market index/options positioning described by SEBI; Jane Street says trades were legitimate arbitrage | SEBI's July 3, 2025 interim order alleges coordinated cash/futures/index-options trading; Jane Street disputes manipulation and characterizes trades as arbitrage. Proceedings on merits remained unresolved in public reporting reviewed through late 2026. citeturn638375search9turn638375news10turn638375search4 | **Alleged / contested, not a confirmed proprietary algorithm** | Bank Nifty cash, futures and options | Not primarily evidence about ultra-low latency | SEBI alleged ₹48.4bn unlawful gains, but this is disputed and should not be treated as established profit from manipulation. citeturn638375search9turn638375search4 |

Key primary URLs requested by the user include:

`https://blog.janestreet.com/flex-model-an-event-driven-feature-engineering-system/`  
`https://www.hudsonrivertrading.com/machine-learning/`  
`https://www.citadelsecurities.com/careers/quantitative-research/`  
`https://www.xtxmarkets.com/`  
`https://www.optiver.com/insights/technology-blog/machine-learning-in-trading-an-adversarial-research-problem/`  
`https://www.jumptrading.com/trading`  
`https://www.jumptrading.com/technology`  
`https://tower-research.com/disclosures-of-fx-trading-practices/`  
`https://www.drw.com/access-our-liquidity/exchange-traded-funds`  
`https://www.flowtraders.com/media/2s2pdl01/flow-traders-annual-report-2025.pdf`  
`https://www.sebi.gov.in/sebi_data/attachdocs/jul-2025/1751584518593.pdf`

## What the algorithms most likely look like

### Market making is prediction plus inventory control

The public evidence strongly suggests that modern HFT market making is not simply “quote bid and ask around the midpoint.” Optiver explicitly says pricing combines many weak signals including trades, order-book movement and related instruments; Jane Street describes event-derived features feeding neural networks; XTX describes ML price forecasts; Citadel describes sophisticated pricing models and real-time quantitative research. citeturn634502search0turn634502search2turn911922search23turn956800search3

A useful academic baseline is Avellaneda–Stoikov, which optimizes market-maker quotes under inventory risk and stochastic fills. The paper derives an inventory-adjusted reservation price and optimal spread rather than simply quoting symmetrically around the mid. It is a foundational model, **not evidence that Jane Street, Citadel, Optiver or another named firm literally deploys it in production**. citeturn655666search23turn655666search2

For inventory \(q_t\), midprice \(S_t\), risk aversion \(\gamma\), volatility \(\sigma\) and horizon \(T\), the canonical reservation price is approximately

\[
r_t=S_t-q_t\gamma\sigma^2(T-t).
\]

With an exponential fill-intensity approximation \(\lambda(\delta)=Ae^{-k\delta}\), a commonly written form of the total optimal quoted spread is

\[
\Delta_t
=
\gamma\sigma^2(T-t)
+\frac{2}{\gamma}\ln\left(1+\frac{\gamma}{k}\right).
\]

A more realistic production-style generalization is

\[
r_t
=
S_t
+\hat\alpha_t
-q_t\Gamma_t
-\text{adverseSelection}_t,
\]

where \(\hat\alpha_t\) is a short-horizon price forecast, \(\Gamma_t\) is dynamic inventory risk, and adverse-selection penalties depend on current book state. This modified equation is an engineering synthesis rather than a claimed firm formula; public statements from Optiver, Jane Street, HRT and XTX make the predictive component plausible, while the risk adjustment is standard market-making economics. citeturn634502search0turn634502search2turn956800search8turn911922search23

A simplified implementation is:

```text
on_market_event(event):

    book = update_order_book(event)

    mid = (best_bid + best_ask) / 2
    alpha = predictive_model(features(book, trades, related_assets))

    inventory_penalty = gamma * inventory * forecast_variance
    adverse_selection = toxicity_model(book, recent_flow)

    fair = mid + alpha - inventory_penalty - adverse_selection

    fill_bid = estimate_fill_probability(BID, candidate_bid, queue_state)
    fill_ask = estimate_fill_probability(ASK, candidate_ask, queue_state)

    expected_bid_value =
        fill_bid * (fair - candidate_bid - fees - expected_postfill_loss)

    expected_ask_value =
        fill_ask * (candidate_ask - fair - fees - expected_postfill_loss)

    quote only if expected value > threshold
    widen / cancel if toxicity or inventory exceeds limits
```

The key insight is that the difficult parts are usually **\(\hat\alpha\), fill probability and post-fill adverse-selection modeling**, not the inventory algebra.

### Order-book imbalance, microprice and short-horizon prediction

Order-book imbalance is one of the most reproducible public microstructure signals. Cont, Kukanov and Stoikov found that short-interval price changes were more robustly related to **order-flow imbalance** than to raw traded volume, with impact inversely related to available depth. citeturn655666academia24

At level one, define bid depth \(Q_b\) and ask depth \(Q_a\):

\[
I_t=\frac{Q_b-Q_a}{Q_b+Q_a}.
\]

A common microprice estimate is

\[
P_{\mu}
=
\frac{P_a Q_b+P_bQ_a}{Q_b+Q_a}.
\]

When bid depth dominates, the microprice lies closer to the ask; when ask depth dominates, it lies closer to the bid. This is useful as a short-horizon fair-value feature, but profitable firms almost certainly combine it with many other signals. Optiver's own account explicitly describes combining hundreds of weak variables rather than relying on one rule, while Jane Street's feature system was designed precisely to compute many event-derived predictors. citeturn634502search0turn634502search2

An implementable linear baseline is

\[
\hat r_{t,t+h}
=
\beta_0
+\beta_1 I_t
+\beta_2 OFI_t
+\beta_3\Delta P_{\text{leader},t}
+\beta_4\,\text{tradeSignEMA}_t
+\beta_5\,\text{spread}_t
+\epsilon_t.
\]

A production model might replace this with boosted trees, neural networks or an ensemble. Jane Street explicitly lists neural nets, random forests, gradient-boosted trees and ensemble methods as ML approaches relevant to its trading environment; that is evidence about its toolset, not proof that any one of these drives a specific instrument. citeturn634502search5

### Cross-asset, ETF and statistical arbitrage

ETF market makers such as Jane Street, DRW, Citadel Securities, IMC and Flow Traders have public evidence of trading both ETFs and their related instruments. Jane Street says it operates in both primary and secondary ETF markets; DRW is an authorized participant for more than 2,000 ETFs; IMC is a lead market maker in more than 150 U.S. ETFs. citeturn956800search0turn709734search7turn911922search0

The generic arbitrage equation is straightforward:

\[
V^{ETF}_t
=
\sum_i w_iP_{i,t}
+C_t-D_t-F_t,
\]

where \(C_t\) accounts for cash components, \(D_t\) distributions/accruals and \(F_t\) funding, transaction, borrow and creation/redemption costs.

Define

\[
B_t=P^{ETF}_t-V^{ETF}_t.
\]

A tradable discrepancy exists only if

\[
|B_t|>
C_{\text{execution}}
+C_{\text{hedge}}
+C_{\text{borrow}}
+C_{\text{latency risk}}
+C_{\text{capital}}.
\]

At institutional scale, the difficult problem is calculating and hedging fair value across thousands of components, stale markets, futures and related ETFs, while deciding whether to create/redeem or warehouse risk. Jane Street's exchange talk even uses computation of an ETF's value as an example of the kind of dependency graph its systems evaluate. citeturn956800search2

For general statistical arbitrage, HRT publicly uses factor models to decompose returns and risk:

\[
\mathbf r_t
=
\mathbf B_t\mathbf f_t+\boldsymbol\epsilon_t,
\]

with covariance

\[
\Sigma_r
=
B\Sigma_fB'
+\Sigma_\epsilon.
\]

HRT explicitly discusses using such structures for portfolio volatility estimation, optimization and return attribution. The public article is a risk/modeling tutorial rather than disclosure of a live HFT alpha, so treating residual mean reversion as HRT's proprietary strategy would go beyond the evidence. citeturn634502search1

For an independently implementable residual strategy:

\[
z_{i,t}
=
\frac{\epsilon_{i,t}-\mu_{\epsilon_i}}
{\sigma_{\epsilon_i}},
\]

and trade only when expected convergence exceeds realistic costs. A simple version can be reproduced without HFT infrastructure at seconds-to-minutes horizons.

### Options market making and volatility arbitrage

Options firms face a much higher-dimensional problem because each underlying can have thousands of strikes and expiries. Citadel Securities says it continuously provides liquidity across over one million option series and sends more than 25 billion option quotes per average day, while IMC describes a valuation library sitting on the real-time critical path of its HFT market-making systems. citeturn956800search3turn911922search14

A minimal options engine starts from a pricing function

\[
C=C(S,K,T,r,q,\sigma)
\]

and estimates Greeks

\[
\Delta=\frac{\partial C}{\partial S},\quad
\Gamma=\frac{\partial^2C}{\partial S^2},\quad
\nu=\frac{\partial C}{\partial\sigma},\quad
\Theta=\frac{\partial C}{\partial t}.
\]

The practical fair value is more realistically based on an implied-volatility surface

\[
\sigma_{\mathrm{impl}}
=
f(K/S,T,\text{rates},\text{dividends},\text{flow},\text{regime}),
\]

followed by bid/ask adjustments for inventory, Greeks, hedge liquidity, toxicity and expected fill value. Optiver explicitly explains implied volatility as a central options-market concept and separately describes dynamic hedging of risk acquired through liquidity provision. citeturn911922search11turn911922search26

DRW publicly recruits for **dispersion trading**, identifying opportunities between index and single-stock options. citeturn448436search13 A generic dispersion relationship is

\[
\sigma_{index}^{2}
=
\sum_i w_i^2\sigma_i^2
+
2\sum_{i<j} w_iw_j\rho_{ij}\sigma_i\sigma_j.
\]

If index implied variance is expensive relative to component implied variances under a realistic correlation assumption, a trader might sell index volatility and buy a delta-hedged basket of component volatility; the opposite trade is possible when the relation reverses. The implementation is dominated by transaction costs, correlation risk, skew/smile differences, discrete hedging and jumps—not merely the algebra.

### Latency arbitrage and stale-price trading

There is strong industry-level evidence that stale-quote arbitrage exists. The SEC's Report to Congress on Algorithmic Trading explicitly discusses HFT firms using speed to trade against stale quotes, while SEC market-data rulemaking cites research finding that relative latency affects HFT performance and can generate adverse selection for slower traders. citeturn709734search12turn709734search13

What is **not** justified is saying “Citadel/Jane Street/Jump use strategy X to pick off stale quotes” without specific evidence. Firms publicly emphasize low latency—Jump describes ASIC/FPGA development, Tower maintains dozens of colocation centers, and Citadel options engineering explicitly cites exchange co-location—but infrastructure capability is not proof of a particular latency-arbitrage tactic. citeturn448436search28turn709734search1turn956800search5

The generic model is a leader-laggard detector:

\[
\hat P^{fair}_{A,t}
=
g(P_{B,t},P_{C,t},F_t,\ldots).
\]

If

\[
P^{ask}_{A,t}
<
\hat P^{fair}_{A,t}
-\text{all-in-cost-buffer},
\]

the quote may be stale and theoretically buyable; similarly for stale bids.

```text
receive fastest available feed from leader markets

fair = cross_asset_model(leader_prices)

for venue in lagging_venues:
    if venue.ask + all_costs < fair - risk_buffer:
        send IOC buy
    if venue.bid - all_costs > fair + risk_buffer:
        send IOC sell

hedge immediately on leader / correlated market
```

In actual competitive markets, this strategy is primarily an **infrastructure race** involving physical distance, feed normalization, FPGA/network processing, exchange gateways and queue timing. The SEC literature notes that advantages can be determined by extremely small relative latency differences rather than absolute speed. citeturn709734search13

### Execution algorithms

Not every profitable algorithm tries to predict price direction. Execution algorithms seek to trade a required quantity while minimizing market impact, spread cost and timing risk. The canonical Almgren–Chriss framework minimizes a combination of expected execution cost and variance of cost under temporary and permanent price impact. citeturn479043search11

A discrete objective can be written as

\[
\min_{\{x_t\}}
E[C]
+\lambda \operatorname{Var}(C),
\qquad
\sum_t x_t=X.
\]

A practical execution policy can condition on volume, spread, imbalance, alpha and impact:

```text
remaining = target_quantity

every decision interval:
    expected_alpha = short_horizon_model()
    liquidity = current_depth_and_volume()
    impact = calibrated_impact_model()
    spread = current_spread()

    urgency =
        risk_aversion * remaining
        + schedule_deficit
        - favorable_alpha_adjustment

    choose:
        passive limit
        aggressive limit
        marketable order
        or wait

    subject to:
        participation cap
        price limits
        venue limits
        inventory / risk limits
```

Citadel Securities publicly offers an algorithmic execution suite and Virtu operates a major execution-services business in addition to market making, but their proprietary routing and scheduling logic is not public. citeturn956800search3turn911922search32

## How to backtest these strategies realistically

The largest difference between a credible HFT backtest and a misleading one is the execution model. Optiver's 2026 discussion is unusually explicit: observing that a price traded historically does **not** mean your hypothetical passive order would have filled, because another participant may have been ahead of you in the queue; moreover, a fill may occur exactly when informed flow is moving price against you. citeturn634502search0

A useful event-driven backtester should reconstruct order-book state from message-level data rather than bars whenever possible:

\[
S_t=
\{P_b,Q_b,P_a,Q_a,L_2,\ldots,L_n,
\text{trades},\text{cancels},\text{adds}\}.
\]

For each hypothetical order, maintain:

\[
Q^{ahead}_{t}
=
Q^{visible}_{price,t}
+
Q^{hiddenEstimate}_{t}
-
Q^{executedAhead}_{t}
-
Q^{cancelledAhead}_{t}.
\]

The order fills only when enough executable volume actually reaches your queue position. Queue valuation has been identified in the literature as an important component of optimal market-making and execution decisions. citeturn655666search17turn655666search18

The strategy's realized P&L should be decomposed as

\[
PnL
=
\text{spread capture}
+\text{mark-to-market alpha}
-\text{fees}
+\text{rebates}
-\text{slippage}
-\text{hedge costs}
-\text{market impact}
-\text{borrow/funding}
-\text{adverse selection}.
\]

Adverse selection should be measured after fills:

\[
AS(h)
=
\operatorname{side}\cdot(P_{t+h}-P_{fill}).
\]

For a bid fill, a negative subsequent move is adverse; for an ask fill, a positive subsequent move is adverse. This post-fill markout is one of the most useful diagnostics for separating genuine spread capture from “being paid the spread to take toxic flow.” Research using live market-making experiments has documented the interaction between queue position, fill probability and post-fill returns. citeturn655666search6

The minimum realistic simulation stack is:

| Component | What to model | Why it matters |
|---|---|---|
| **Market data latency** | Feed delay + normalization delay | Your model cannot trade on information that has not reached it. SEC research and firm disclosures make relative latency economically material. citeturn709734search13turn956800search5 |
| **Decision latency** | Feature/model calculation time | Neural or complex models may produce better forecasts but miss fills if slower. |
| **Order-entry latency** | Gateway/network/exchange delay | Determines whether a predicted opportunity still exists. |
| **Queue position** | Shares/contracts ahead, cancellations, executions | Price-level trades do not imply your order filled. citeturn634502search0turn655666search17 |
| **Maker/taker fees** | Actual venue schedule | A one-tick edge can disappear after fees. |
| **Rebates** | Venue-specific | Can materially alter passive-order economics. |
| **Slippage** | Depth consumed at each level | Necessary for aggressive execution. |
| **Market impact** | Temporary + permanent response to size | Almgren–Chriss explicitly treats both execution cost and volatility risk. citeturn479043search11 |
| **Adverse selection** | Post-fill markouts | Essential for market making. citeturn634502search0turn709734search12 |
| **Hedging latency/cost** | Underlying/future/ETF/options hedge | Especially important for derivatives and cross-market trading. |
| **Inventory** | Position limits, Greeks, factor exposures | Market makers cannot assume infinite balance sheet. |
| **Message limits** | Exchange throttle/order-to-trade rules | Real systems cannot cancel infinitely fast. |
| **Data leakage** | Strict event timestamp ordering | Optiver explicitly warns that future data entering feature construction invalidates backtests. citeturn634502search0 |
| **Strategy footprint** | Your own order changes book state | Historical replay does not contain the strategy's counterfactual impact. Optiver specifically highlights this limitation. citeturn634502search0 |

A useful test architecture separates **signal quality** from **monetizability**. First evaluate whether

\[
E[r_{t+h}\mid x_t]
\]

is statistically predictable. Then independently estimate

\[
EV_{\text{trade}}
=
P(fill)\times
(E[\text{gross edge}\mid fill]
-\text{adverse selection})
-\text{fees}
-\text{impact}.
\]

Many research signals survive the first test and fail the second.

For machine learning, use rolling or expanding-window training rather than random train/test splits because the environment is nonstationary. HRT and Optiver both explicitly emphasize distribution shift, adversarial adaptation and model decay. citeturn956800search8turn634502search0

A production-grade walk-forward protocol would therefore train on period \(T_0:T_1\), validate on \(T_1:T_2\), freeze parameters, trade/simulate \(T_2:T_3\), roll forward and repeat. Hyperparameters should be selected only from earlier data, and the final economic evaluation must use net P&L after all simulated costs.

## What an independent researcher can realistically reproduce

The publicly available evidence points to a major distinction between **research edge** and **infrastructure edge**. Firms such as Jump and Tower explicitly invest in FPGA/ASIC systems, specialized networking and colocation; Citadel's options engineers describe hardware constraints created by colocated latency requirements. citeturn448436search28turn709734search1turn956800search5 An independent trader on ordinary internet infrastructure therefore should not expect to compete for the exact same microsecond opportunities.

A practical feasibility ranking is:

| Rank | Strategy family | Independent feasibility | Why |
|---:|---|---|---|
| **1** | **Cross-sectional / factor statistical arbitrage at seconds-to-hours horizons** | **High** | Requires good data and research rather than winning a microsecond race. HRT's public factor methodology provides a useful framework, though not a complete trading strategy. citeturn634502search1 |
| **2** | **ETF / futures relative-value at slower horizons** | **Medium-high** | Fair-value relationships are replicable, but true creation/redemption and institutional block pricing require AP relationships and balance sheet. Jane Street and DRW demonstrate the institutional version. citeturn956800search0turn709734search7 |
| **3** | **Options volatility relative value / dispersion** | **Medium** | Mathematical models are accessible; data quality, transaction costs, margin and hedging are harder. DRW publicly confirms dispersion as a real professional strategy family. citeturn448436search13 |
| **4** | **Order-book imbalance prediction at 100ms–seconds** | **Medium** | Research signal is reproducible from depth data; successful execution requires realistic queue/fill modeling. Cont et al. provide strong evidence that order-flow imbalance contains short-horizon information. citeturn655666academia24 |
| **5** | **Passive market making on less competitive venues/products** | **Medium-low** | Can be implemented mathematically, but the independent trader faces adverse selection, queue disadvantage and capital constraints. citeturn655666search23turn634502search0 |
| **6** | **Short-horizon ML market making in major futures/equities** | **Low** | The model can be reproduced conceptually; production economics depend on fast feeds, queue priority and execution engineering. Firms like HRT, Optiver and IMC combine research with low-latency infrastructure. citeturn956800search8turn634502search0turn448436search1 |
| **7** | **True stale-quote / latency arbitrage in major electronic markets** | **Very low** | Relative latency is the core competitive variable, and institutional firms use FPGA, specialized networks and colocation. citeturn709734search13turn448436search28turn709734search1 |

The best independent research target is therefore usually not “copy Citadel's HFT algorithm,” because the proprietary parameterization and execution infrastructure are unavailable. A more defensible project is to reproduce the **economic mechanisms** visible in public sources: order-flow prediction, cross-asset fair value, inventory-aware quoting, volatility-surface relative value, factor-neutral residual trading and execution-cost optimization. These mechanisms are demonstrably relevant to the firms studied, while the exact production signals remain private. citeturn634502search0turn634502search2turn911922search23turn448436search1

## Bottom line

The strongest public evidence supports a common modern HFT stack:

\[
\boxed{
\text{Market data}
\rightarrow
\text{event features}
\rightarrow
\text{short-horizon fair-value forecast}
\rightarrow
\text{inventory/risk adjustment}
\rightarrow
\text{venue + queue model}
\rightarrow
\text{orders}
\rightarrow
\text{hedging}
}
\]

Jane Street, HRT, XTX, Optiver, IMC, Jump and Citadel Securities all disclose pieces of this architecture directly, with varying levels of specificity. citeturn634502search2turn956800search8turn911922search23turn634502search0turn448436search1turn448436search35turn634502search3

The alpha layer increasingly appears to be **statistical or machine-learned prediction from many weak signals**, rather than a single deterministic rule. Jane Street explicitly confirms heavy neural-network use; HRT describes deep learning as core; XTX says ML produces price forecasts across tens of thousands of instruments; Optiver describes ML combining hundreds of weak microstructure and cross-instrument signals; IMC and Jump both publicly recruit and build around predictive ML. citeturn634502search2turn956800search8turn911922search23turn634502search0turn448436search2turn448436search35

The execution layer is equally important. Queue position, latency, fees and adverse selection can turn an apparently profitable signal into a losing strategy. Optiver's public description of this problem closely matches the academic literature on queue valuation and the SEC's evidence on speed and stale-price adverse selection. citeturn634502search0turn655666search17turn709734search12turn709734search13

Finally, “HFT profitability” should not be conflated with pure speed arbitrage. DRW publicly describes both latency-sensitive trading and ETF quoting at hundreds-of-milliseconds response times; XTX emphasizes holding risk for meaningful periods; Jane Street and HRT operate businesses that extend beyond ultra-short-horizon market making. citeturn448436search16turn709734search7turn911922search41turn470137news50 The publicly observable competitive advantage of the best firms is therefore better understood as the combination of **prediction + pricing + execution + infrastructure + capital + risk management**, rather than one secret formula.

None of the models above guarantees positive returns. The academic formulas are useful starting points; the public firm evidence shows what broad families are genuinely used; and the gap between those two is precisely where the proprietary edge of the profitable HFT firms resides.