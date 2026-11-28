"""Prometheus text-format metrics with the standard library only.

The idcard-detector and movie-recommender servers run on the ultralytics image
as it is, which has no prometheus_client, and they are not rebuilt for this.
This module covers what they need: counters, gauges and histograms with
labels, rendered in the text exposition format 0.0.4 that Prometheus scrapes
from GET /metrics. Each ConfigMap mounts it next to server.py in /app, which
is on sys.path because the container runs python3 /app/server.py.

One process, threads only (ThreadingHTTPServer): one lock guards every update
and the render.
"""
import math
import threading

CONTENT_TYPE = 'text/plain; version=0.0.4; charset=utf-8'


def _escape(value):
    return str(value).replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"')


def _labels(names, values, extra=()):
    pairs = [*zip(names, values), *extra]
    if not pairs:
        return ''
    return '{' + ','.join(f'{n}="{_escape(v)}"' for n, v in pairs) + '}'


def _number(value):
    if value == math.inf:
        return '+Inf'
    return repr(float(value))


class _Metric:
    kind = ''

    def __init__(self, registry, name, documentation, labelnames):
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self._lock = registry.lock
        self._series = {}

    def _key(self, values):
        if len(values) != len(self.labelnames):
            raise ValueError(f'{self.name} takes labels {self.labelnames}, got {values}')
        return tuple(str(v) for v in values)

    def exposed_name(self):
        return self.name

    def render(self):
        name = self.exposed_name()
        lines = [f'# HELP {name} {self.documentation}', f'# TYPE {name} {self.kind}']
        for key in sorted(self._series):
            lines.extend(self._render_series(key, self._series[key]))
        return lines


class Counter(_Metric):
    kind = 'counter'

    def exposed_name(self):
        # Format 0.0.4 names the family with the suffix too (prometheus_client does the same);
        # without it Prometheus reads the _total samples as untyped.
        return f'{self.name}_total'

    def labels(self, *values):
        """Create the series at 0 so a label that has not happened yet still shows."""
        with self._lock:
            self._series.setdefault(self._key(values), 0.0)

    def inc(self, *values, amount=1.0):
        with self._lock:
            key = self._key(values)
            self._series[key] = self._series.get(key, 0.0) + amount

    def _render_series(self, key, value):
        return [f'{self.name}_total{_labels(self.labelnames, key)} {_number(value)}']


class Gauge(_Metric):
    kind = 'gauge'

    def set(self, value, *values):
        with self._lock:
            self._series[self._key(values)] = float(value)

    def _render_series(self, key, value):
        return [f'{self.name}{_labels(self.labelnames, key)} {_number(value)}']


class Histogram(_Metric):
    kind = 'histogram'

    def __init__(self, registry, name, documentation, labelnames, buckets):
        super().__init__(registry, name, documentation, labelnames)
        self.buckets = tuple(sorted(float(b) for b in buckets)) + (math.inf,)

    def labels(self, *values):
        with self._lock:
            self._new(self._key(values))

    def _new(self, key):
        return self._series.setdefault(key, {'counts': [0] * len(self.buckets), 'sum': 0.0})

    def observe(self, value, *values):
        value = float(value)
        with self._lock:
            series = self._new(self._key(values))
            # Cumulative buckets: one observation counts in every bucket whose bound it does not exceed.
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    series['counts'][i] += 1
            series['sum'] += value

    def _render_series(self, key, series):
        lines = [f'{self.name}_bucket{_labels(self.labelnames, key, [("le", _number(b))])} {c}'
                 for b, c in zip(self.buckets, series['counts'])]
        lines.append(f'{self.name}_sum{_labels(self.labelnames, key)} {_number(series["sum"])}')
        lines.append(f'{self.name}_count{_labels(self.labelnames, key)} {series["counts"][-1]}')
        return lines


class Registry:
    def __init__(self):
        self.lock = threading.Lock()
        self._metrics = []

    def _add(self, metric):
        self._metrics.append(metric)
        return metric

    def counter(self, name, documentation, labelnames=()):
        return self._add(Counter(self, name, documentation, labelnames))

    def gauge(self, name, documentation, labelnames=()):
        return self._add(Gauge(self, name, documentation, labelnames))

    def histogram(self, name, documentation, labelnames=(), buckets=(0.5, 1, 2.5, 5, 10)):
        return self._add(Histogram(self, name, documentation, labelnames, buckets))

    def render(self):
        with self.lock:
            lines = [line for metric in self._metrics for line in metric.render()]
        return ('\n'.join(lines) + '\n').encode()


# Request latency of one predict call, in seconds. CPU inference on the shared
# node: YOLO takes a few hundred ms per image, a recommendation a few ms.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
