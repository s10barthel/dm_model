"""Ordered preparation with one window, ready or in progress, ahead of a consumer."""
from __future__ import annotations

from threading import Condition, Thread
import time


def interval_overlap(work, compute):
    """Wall time shared by disjoint preparation and disjoint compute intervals."""
    return sum(max(0.0, min(end, right) - max(start, left))
               for start, end in work for left, right in compute)


class RuntimePreparation:
    """The consumer owns publication; the producer exclusively advances the iterator.

    A producer needs permission before calling next(), not merely before enqueueing
    its result. Consequently a full slot cannot hide another window under construction.
    Cancellation waits for the current next() call, whose logical groups stay intact.
    """

    def __init__(self, iterator, *, enabled=False):
        self.iterator = iter(iterator)
        self.requested_prefetch = self.enabled = bool(enabled)
        self._condition = Condition()
        self._requested = True
        self._stopped = False
        self._slot = None
        self._thread = None
        self._finished = False
        self._closed = False
        self.work = []
        self.consumer_wait_seconds = 0.0
        self.initial_wait_seconds = 0.0
        self.producer_backpressure_seconds = 0.0
        self.shutdown_seconds = 0.0
        self.peak_prepared_rows = 0
        self.delivered = 0

    def __enter__(self):
        if self.enabled:
            self._thread = Thread(target=self._produce, name="pc-xpass-preparation", daemon=False)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def __iter__(self):
        return self

    def _advance(self):
        started = time.perf_counter()
        try:
            value = next(self.iterator)
            rows = int(value.get("row_count", 0))
            self.peak_prepared_rows = max(self.peak_prepared_rows, rows)
            result = ("value", value)
        except StopIteration:
            result, rows = ("end", None), 0
        except BaseException as exc:
            result, rows = ("error", exc), 0
        finished = time.perf_counter()
        self.work.append(dict(started=started, finished=finished,
                              seconds=finished - started, rows=rows, kind=result[0]))
        return result

    def _produce(self):
        while True:
            with self._condition:
                waiting = time.perf_counter()
                self._condition.wait_for(lambda: self._stopped or self._requested)
                self.producer_backpressure_seconds += time.perf_counter() - waiting
                if self._stopped:
                    return
                self._requested = False
            result = self._advance()
            with self._condition:
                self._slot = result
                self._condition.notify_all()
            if result[0] != "value":
                return

    def __next__(self):
        if self._closed or self._finished:
            raise StopIteration
        if self._thread is not None:
            started = time.perf_counter()
            with self._condition:
                self._condition.wait_for(lambda: self._slot is not None)
                result, self._slot = self._slot, None
                self._requested = True
                self._condition.notify_all()
            elapsed = time.perf_counter() - started
            self.consumer_wait_seconds += elapsed
            if not self.delivered:
                self.initial_wait_seconds += elapsed
        elif self._slot is not None:
            result, self._slot = self._slot, None
        else:
            result = self._advance()
        kind, value = result
        if kind != "value":
            self._finished = True
            if kind == "error":
                raise value
            raise StopIteration
        self.delivered += 1
        return value

    def stop_prefetch(self):
        """Join before recovery; preserve the ready result and resume sequentially."""
        if self._thread is None:
            return
        started = time.perf_counter()
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join()
        self._thread = None
        self.enabled = False
        self.shutdown_seconds += time.perf_counter() - started

    def close(self):
        if self._closed:
            return
        self.stop_prefetch()
        try:
            close = getattr(self.iterator, "close", None)
            if close is not None:
                close()
        finally:
            self._slot = None
            self._closed = True

    def summary(self, compute_intervals=()):
        intervals = [(row["started"], row["finished"]) for row in self.work]
        return dict(prefetch_enabled=self.requested_prefetch,
                    preparation_work_seconds=sum(row["seconds"] for row in self.work),
                    consumer_wait_seconds=self.consumer_wait_seconds,
                    initial_wait_seconds=self.initial_wait_seconds,
                    producer_backpressure_seconds=self.producer_backpressure_seconds,
                    observed_compute_overlap_seconds=interval_overlap(intervals, compute_intervals),
                    shutdown_seconds=self.shutdown_seconds,
                    peak_ahead_windows=int(self.requested_prefetch and bool(self.work)),
                    peak_prepared_window_rows=self.peak_prepared_rows,
                    work=list(self.work))
