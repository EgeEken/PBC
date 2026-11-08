import time
from collections import defaultdict
from contextlib import contextmanager


class TimingTrace:
    """Low-overhead, opt-in nested timing trace for one codec operation."""

    def __init__(self, operation: str, metadata: dict | None = None):
        self.operation = operation
        self.metadata = dict(metadata or {})
        self.started_ns = time.perf_counter_ns()
        self.started_wall = time.time()
        self.events = []
        self.counters = defaultdict(int)
        self._stack = []
        self._finished_ns = None

    def count(self, name: str, amount: int = 1) -> None:
        self.counters[name] += int(amount)

    @contextmanager
    def span(self, name: str, **metadata):
        event = {
            "name": name,
            "start_ns": time.perf_counter_ns() - self.started_ns,
            "duration_ns": 0,
            "parent": self._stack[-1] if self._stack else None,
            "metadata": metadata,
        }
        index = len(self.events)
        self.events.append(event)
        self._stack.append(index)
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            event["duration_ns"] = time.perf_counter_ns() - started
            self._stack.pop()

    def finish(self):
        if self._finished_ns is None:
            self._finished_ns = time.perf_counter_ns()

    def report(self, metadata: dict | None = None, *, finish: bool = True) -> dict:
        if finish:
            self.finish()
        end_ns = self._finished_ns if self._finished_ns is not None else time.perf_counter_ns()
        total_ns = end_ns - self.started_ns
        grouped = defaultdict(list)
        top_level_ns = 0
        child_ns = defaultdict(int)
        for event in self.events:
            if event["parent"] is not None:
                child_ns[event["parent"]] += event["duration_ns"]
        for index, event in enumerate(self.events):
            event["exclusive_ns"] = max(0, event["duration_ns"] - child_ns[index])
        for event in self.events:
            grouped[event["name"]].append(event)
            if event["parent"] is None:
                top_level_ns += event["duration_ns"]
        summary = {}
        for name, events in sorted(grouped.items()):
            values = [event["duration_ns"] for event in events]
            exclusive = [event["exclusive_ns"] for event in events]
            summary[name] = {
                "count": len(values),
                "total_ns": int(sum(values)),
                "exclusive_ns": int(sum(exclusive)),
                "avg_ns": int(sum(values) / len(values)),
                "min_ns": int(min(values)),
                "max_ns": int(max(values)),
            }
        merged_metadata = {**self.metadata, **(metadata or {})}
        return {
            "schema_version": 1,
            "operation": self.operation,
            "started_wall": self.started_wall,
            "total_ns": int(total_ns),
            "top_level_ns": int(top_level_ns),
            "unattributed_ns": int(max(0, total_ns - top_level_ns)),
            "metadata": merged_metadata,
            "counters": dict(sorted(self.counters.items())),
            "events": self.events,
            "summary": summary,
        }


def timed(trace, name: str, **metadata):
    return trace.span(name, **metadata) if trace is not None else _NullSpan()


class _NullSpan:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False
