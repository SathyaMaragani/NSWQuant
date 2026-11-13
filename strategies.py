"""Strategies for hftsim.py. Add your own: subclass hftsim.Strategy and use the env API.

env (an hftsim.Exchange) gives you, at every callback:
  env.bp, env.bq, env.ap, env.aq   top 5 bid/ask prices (integer ticks) and sizes, kept current between snapshots
  env.mid, env.pos, env.orders     mid (ticks), position (base units), your open orders {oid: Order}
  env.signal                       (expected move in ticks, confident-down flag, confident-up flag, class probabilities)
  env.place(side, px, qty)         post-only limit order, +1 buy / -1 sell, returns order id (None if rate-capped)
  env.cancel(oid), env.market(side, qty)
Callbacks: on_book(env) every book update; on_trade(env, px, qty, aggressor) every trade (only with a fast profile);
on_fill(env, fill).
"""
import math

from hftsim import FAR, Strategy

FAR_AHEAD = 1e18


class QuotingMM(Strategy):
    """Inventory-aware market maker (Avellaneda-Stoikov idea, no model):
    fair = mid - inv_skew * position; bid = floor(fair - half), ask = ceil(fair + half)."""

    def __init__(self, size, half=0.5, inv_skew=0.4, qmax=5, kappa=0.0, use_signal=False, pull=False,
                 guard=False, guard_mult=2.0):
        self.size, self.half, self.inv_skew, self.qmax = size, half, inv_skew, qmax
        self.kappa, self.use_signal, self.pull = kappa, use_signal, pull
        self.guard, self.guard_mult = guard, guard_mult
        self.quote = {+1: None, -1: None}          # side -> [oid, px]

    def targets(self, env):
        fair = env.mid - self.inv_skew * env.pos / self.size
        move, confident_down, confident_up, _ = env.signal
        if self.use_signal:
            fair += self.kappa * move
        bid = min(math.floor(fair - self.half), env.ap[0] - 1)
        ask = max(math.ceil(fair + self.half), env.bp[0] + 1)
        if env.pos >= self.qmax * self.size or (self.pull and confident_down):
            bid = None
        if env.pos <= -self.qmax * self.size or (self.pull and confident_up):
            ask = None
        return {+1: bid, -1: ask}

    def on_book(self, env):
        for side, px in self.targets(env).items():
            cur = self.quote[side]
            alive = cur is not None and cur[0] in env.orders
            if alive and cur[1] == px:
                continue                           # unchanged: keep queue priority
            if alive and px is None:
                if env.cancel(cur[0]):             # if rate-capped, keep tracking the order and retry next time
                    self.quote[side] = None
            elif alive:
                new = env.replace(cur[0], px)      # amend: one message, never leaves an orphan order
                if new:
                    self.quote[side] = [new, px]
            elif px is not None:
                oid = env.place(side, px, self.size)
                self.quote[side] = [oid, px] if oid else None
            else:
                self.quote[side] = None

    def on_trade(self, env, px, qty, aggressor):
        """Queue-depletion guard (needs a fast cancel): when trades have nearly emptied the level we are resting on,
        the price is about to move through it, so step away instead of being the last fill."""
        if not self.guard:
            return
        side = -aggressor
        cur = self.quote[side]
        if cur is None or cur[0] not in env.orders or cur[1] != px:
            return
        left = env.bq[0] if side > 0 and env.bp[0] == px else env.aq[0] if side < 0 and env.ap[0] == px else None
        if left is not None and left <= self.guard_mult * self.size and env.cancel(cur[0]):
            self.quote[side] = None

    def on_fill(self, env, fill):
        for side in (+1, -1):
            cur = self.quote[side]
            if cur is not None and cur[0] == fill.oid and fill.oid not in env.orders:
                self.quote[side] = None            # fully filled: re-quote on the next book update


class NSQMM(QuotingMM):
    """NSQ-MM: the market maker above, steered by the TinyML model.
    fair += kappa * E[move]; pull the bid on a confident DOWN call and the ask on a confident UP call."""

    def __init__(self, size, half=0.5, inv_skew=0.4, kappa=1.0, guard=False, **kw):
        super().__init__(size, half=half, inv_skew=inv_skew, kappa=kappa, use_signal=True, pull=True, guard=guard, **kw)


class QueueAwareMM(NSQMM):
    """NSQ-MM-Q: NSQ-MM that only takes fills it expects to win. From the per-fill diagnosis (day-1 data):
    fills from the back of a queue only happen when a sweep runs through the level, and fills the model saw
    coming lose. So, per side:
      - rest only near the front of the queue: an order with more than max_ahead lots ahead is cancelled and
        that price is skipped until the target moves;
      - quote only if the model's probability of a move against that side is <= tau;
      - at the touch, quote only if our side holds at least imb_min of the touch depth (a thin side gets run over)."""

    def __init__(self, size, tau=1.0, imb_min=0.0, max_ahead=FAR_AHEAD, **kw):
        super().__init__(size, **kw)
        self.tau, self.imb_min, self.max_ahead = tau, imb_min, max_ahead
        self.blocked = {+1: None, -1: None}        # side -> price we left because the queue was too long

    def targets(self, env):
        t = super().targets(env)
        prob = env.signal[3]
        for side in (+1, -1):
            px = t[side]
            if px is None:
                continue
            if px != self.blocked[side]:
                self.blocked[side] = None
            p_adv = prob[0] if side > 0 else prob[2]
            own, opp = (env.bq[0], env.aq[0]) if side > 0 else (env.aq[0], env.bq[0])
            at_touch = px == (env.bp[0] if side > 0 else env.ap[0])
            if p_adv > self.tau or px == self.blocked[side] or (at_touch and own < self.imb_min * (own + opp)):
                t[side] = None
        return t

    def _check_queue(self, env):
        for side in (+1, -1):
            cur = self.quote[side]
            o = env.orders.get(cur[0]) if cur else None
            if o is not None and o.live and o.queue_ahead > self.max_ahead * self.size and env.cancel(o.oid):
                self.blocked[side], self.quote[side] = o.px, None

    def on_book(self, env):
        self._check_queue(env)
        super().on_book(env)

    def on_trade(self, env, px, qty, aggressor):
        super().on_trade(env, px, qty, aggressor)
        self._check_queue(env)
        if -FAR < env.bp[0] and env.ap[0] < FAR:   # a sweep can empty one side of the shadow book until the next snapshot
            self.on_book(env)                      # re-quote on every trade: fast infrastructure only


class ImbalanceMM(QuotingMM):
    """Port of hftbacktest's "Market Making with Alpha - Order Book Imbalance" (github.com/nkaz001/hftbacktest, MIT).
    alpha = z-score of (total bid size - total ask size) across the published book, sampled once a second over a
    rolling window (env.aux[zcol], precomputed causally by the runner);
    fair = mid + c1 * alpha;  reservation = fair - skew * position / order size;
    bid = min(reservation - half, best bid), ask = max(reservation + half, best ask); requote once per interval_ms;
    stop quoting a side at max_pos lots. Prices are in basis points of the mid so one setting fits ETH and BTC."""

    def __init__(self, size, half_bp=0.5, c1_bp=0.5, skew_bp=0.05, zcol=0, interval_ms=1000, max_pos=20, **kw):
        super().__init__(size, qmax=max_pos, **kw)
        self.half_bp, self.c1_bp, self.skew_bp, self.zcol = half_bp, c1_bp, skew_bp, int(zcol)
        self.interval_ms, self.max_pos, self.next_t = interval_ms, max_pos, 0

    def targets(self, env):
        tpb = env.mid * 1e-4                       # ticks per basis point (mid is in ticks)
        alpha = env.aux[self.zcol] if env.aux is not None else 0.0
        alpha = alpha if math.isfinite(alpha) else 0.0
        res = env.mid + self.c1_bp * tpb * alpha - self.skew_bp * tpb * env.pos / self.size
        bid = min(math.floor(res - self.half_bp * tpb), env.bp[0])
        ask = max(math.ceil(res + self.half_bp * tpb), env.ap[0])
        return {+1: None if env.pos >= self.max_pos * self.size else bid,
                -1: None if env.pos <= -self.max_pos * self.size else ask}

    def on_book(self, env):
        if env.now < self.next_t:
            return
        self.next_t = env.now + self.interval_ms
        super().on_book(env)


class SignalSwing(Strategy):
    """D: a slow directional strategy for a retail account (no rebate, back of the queue, 50 ms).
    When the Binance +-2.5 % depth-imbalance z is beyond +-k_in, buy (z > 0) or sell with a post-only order resting at
    the best price, re-pegged to the touch once a second; give up after entry_s. Hold for hold_s, or exit early if the
    signal turns the other way (z * side < -k_out). Exit with a post-only order at the opposite touch; after exit_s
    without a full fill, close the rest at market. One position at a time; cooldown_s before the next entry."""

    def __init__(self, size, k_in=1.5, hold_s=1800, entry_s=120, exit_s=60, k_out=1.0, cooldown_s=60, zcol=2, peg_ms=1000,
                 extend=False):
        self.size, self.k_in, self.hold, self.entry, self.exit_t = size, k_in, hold_s * 1000, entry_s * 1000, exit_s * 1000
        self.extend = extend                       # at the end of the hold, stay in (another hold) while |z| >= k_in our way
        self.k_out, self.cool, self.zcol, self.peg = k_out, cooldown_s * 1000, int(zcol), peg_ms
        self.state, self.side, self.oid, self.t0, self.next_peg, self.free_at = "flat", 0, None, 0, 0, 0
        self.trips = self.market_exits = 0

    def _z(self, env):
        z = env.aux[self.zcol] if env.aux is not None else math.nan
        return z if math.isfinite(z) else math.nan

    def _peg(self, env, side, qty):
        """Keep one post-only order at our side's best price."""
        px = env.bp[0] if side > 0 else env.ap[0]
        o = env.orders.get(self.oid) if self.oid is not None else None
        if o is None:
            self.oid = env.place(side, px, qty)
        elif o.px != px and o.live:
            new = env.replace(self.oid, px)
            if new:
                self.oid = new

    def _cancel(self, env):
        if self.oid is not None and self.oid in env.orders:
            env.cancel(self.oid)
        self.oid = None

    def on_book(self, env):
        now, z = env.now, self._z(env)
        if self.state == "flat":
            if now >= self.free_at and abs(z) >= self.k_in:
                self.side, self.state, self.t0, self.next_peg = (1 if z > 0 else -1), "entering", now, now
        if self.state == "entering":
            if abs(env.pos) > 1e-12:                   # (partly) filled: hold what we got
                self._cancel(env)
                self.state, self.t0 = "holding", now
            elif now - self.t0 > self.entry or not (z * self.side >= self.k_in):
                self._cancel(env)                      # signal gone or waited too long
                self.state = "flat"
                return
            elif now >= self.next_peg:
                self.next_peg = now + self.peg
                self._peg(env, self.side, self.size)
        if self.state == "holding":
            if z * self.side < -self.k_out:
                self.state, self.t0, self.next_peg = "exiting", now, now
            elif now - self.t0 >= self.hold:
                if self.extend and z * self.side >= self.k_in:
                    self.t0 = now                  # still a strong call our way: holding beats paying to exit and re-enter
                else:
                    self.state, self.t0, self.next_peg = "exiting", now, now
        if self.state == "exiting":
            if abs(env.pos) < 1e-12:
                self._cancel(env)
                self.state, self.free_at = "flat", now + self.cool
                self.trips += 1
            elif now - self.t0 > self.exit_t:
                self._cancel(env)
                env.market(-1 if env.pos > 0 else 1, abs(env.pos))
                self.market_exits += self.next_peg < 10 ** 18
                self.t0 = now - self.exit_t + 1000      # if anything is still open in 1 s, take it again
                self.next_peg = 10 ** 18                # no more passive exit orders
            elif now >= self.next_peg:
                self.next_peg = now + self.peg
                self._peg(env, -1 if env.pos > 0 else 1, abs(env.pos))


class SignalTaker(Strategy):
    """Cross the spread on the model's confident calls, hold `hold` book updates, cross back."""

    def __init__(self, size, hold=30):
        self.size, self.hold = size, hold
        self.state, self.count, self.wait = "flat", 0, 0

    def on_book(self, env):
        _, confident_down, confident_up, _ = env.signal
        if self.state == "flat":
            if confident_up or confident_down:
                env.market(+1 if confident_up else -1, self.size)
                self.state, self.wait = "entering", 0
        elif self.state == "entering":
            self.wait += 1
            if abs(env.pos) > 1e-12:
                self.state, self.count = "holding", 0
            elif self.wait > 20:
                self.state = "flat"                # order never filled (empty book side): give up
        elif self.state == "holding":
            self.count += 1
            if self.count >= self.hold:
                env.market(-1 if env.pos > 0 else +1, abs(env.pos))
                self.state = "exiting"
        elif self.state == "exiting" and abs(env.pos) < 1e-12:
            self.state = "flat"
