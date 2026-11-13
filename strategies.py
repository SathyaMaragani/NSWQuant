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

from hftsim import Strategy


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
