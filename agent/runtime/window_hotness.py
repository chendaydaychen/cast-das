"""Paper §5.1: top-q access frequency over a causal sliding window.

Lifetime contention counters remain diagnostics, not alternative hot triggers.
Ties use object-key order only to make the top-q membership deterministic.
No trace priming or knowledge of future keys is accepted.
"""
import collections
import heapq
import math
import threading
import time

from .hotness import HotnessTracker


class WindowFrequencyHotnessTracker(HotnessTracker):
    def __init__(self, *, window_ms=1000., hot_fraction=.10, rank_refresh_ms=10.,
                 clock_ns=time.monotonic_ns):
        super().__init__()
        if not math.isfinite(window_ms) or window_ms<=0:
            raise ValueError('Positive finite hotspot window required')
        if not math.isfinite(hot_fraction) or not 0<hot_fraction<=1:
            raise ValueError('Hot fraction must be in (0,1]')
        if not math.isfinite(rank_refresh_ms) or rank_refresh_ms<0:
            raise ValueError('Nonnegative finite ranking refresh required')
        self.window_ns=int(window_ms*1e6)
        self.hot_fraction=float(hot_fraction)
        self.rank_refresh_ns=int(rank_refresh_ms*1e6)
        self._clock_ns=clock_ns
        self._window_lock=threading.RLock()
        self._events=collections.deque()
        self._counts=collections.Counter()
        self._ranked_at=None
        self._hot=frozenset()

    def _expire(self, now):
        expired_hot=[]
        while self._events and self._events[0][0]<now-self.window_ns:
            _,key=self._events.popleft()
            self._counts[key]-=1
            if not self._counts[key]:
                del self._counts[key]
                if key in self._hot:expired_hot.append(key)
        # Expired objects never remain hot even within the ranking cache.
        # Only zero-count removals can invalidate membership. Preserve the
        # immutable snapshot, but do not copy/intersect the entire hot set on
        # every lookup when nothing relevant expired.
        if expired_hot:self._hot=self._hot.difference(expired_hot)

    def _rank(self, now):
        self._expire(now)
        if self._ranked_at is None or now-self._ranked_at>=self.rank_refresh_ns:
            count=math.ceil(len(self._counts)*self.hot_fraction)
            self._hot=frozenset(heapq.nsmallest(count,self._counts,
                key=lambda key:(-self._counts[key],key)))
            self._ranked_at=now
        return self._hot

    def observe_access(self, object_id):
        key=str(object_id)
        super().observe_access(key)
        with self._window_lock:
            now=self._clock_ns();self._expire(now)
            self._events.append((now,key));self._counts[key]+=1
            return key in self._rank(now)

    def is_hot(self, object_id):
        with self._window_lock:
            return str(object_id) in self._rank(self._clock_ns())

    def hot_targets(self, object_ids):
        with self._window_lock:
            return set(map(str,object_ids)).intersection(self._rank(self._clock_ns()))

    def object_snapshot(self, object_id):
        row=super().object_snapshot(object_id)
        with self._window_lock:
            hot=self._rank(self._clock_ns())
            row.update(hot=str(object_id) in hot,
                window_accesses=self._counts.get(str(object_id),0))
        return row

    def snapshot(self):
        row=super().snapshot()
        with self._window_lock:
            hot=self._rank(self._clock_ns())
            row.update(hot_objects=len(hot),window_observed_objects=len(self._counts),
                window_accesses=sum(self._counts.values()),
                classification='sliding-window-frequency-top-q',
                window_ms=self.window_ns/1e6,hot_fraction=self.hot_fraction,
                rank_refresh_ms=self.rank_refresh_ns/1e6,tie_break='lexical-object-key')
        return row

    def prime_accesses(self, object_ids):
        raise ValueError('Paper frequency window forbids offline trace priming')

    def prime_transaction(self, object_ids):
        raise ValueError('Paper frequency window forbids planned access priming')
