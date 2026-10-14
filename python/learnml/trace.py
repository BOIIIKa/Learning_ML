"""Training traces: the shared format between Python training code and the web player.

Every algorithm — a student's hand-written one or a library run — records one state
per step; the site's trace player renders the same JSON regardless of where it came from.

    trace = Trace("classifier2d", X, y, title="My SMO")
    trace.record(w=w, b=b, highlight=[i, j], metrics={"dual": obj})
    trace.to_dict()
"""

import json

import numpy as np


def _plain(value):
    """Convert numpy scalars / arrays (possibly nested) into JSON-friendly Python values."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class Trace:
    """An ordered list of training states over a fixed 2-D dataset."""

    def __init__(self, kind, X, y, title=None, grid=None):
        self.kind = kind
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.title = title
        self.grid = grid
        self.steps = []

    def record(self, **state):
        """Append one step. Common keys: w, b, field, weights, highlight, support, metrics, note."""
        self.steps.append(_plain(state))

    def to_dict(self):
        out = {"kind": self.kind, "X": _plain(self.X), "y": _plain(self.y), "steps": self.steps}
        if self.title is not None:
            out["title"] = self.title
        if self.grid is not None:
            out["grid"] = {"x": _plain(self.grid.xs), "y": _plain(self.grid.ys)}
        return out

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


class Grid:
    """A regular grid covering a 2-D dataset, for recording decision fields."""

    def __init__(self, X, n=60, pad=0.6):
        X = np.asarray(X, dtype=float)
        lo, hi = X.min(axis=0) - pad, X.max(axis=0) + pad
        self.xs = np.linspace(lo[0], hi[0], n)
        self.ys = np.linspace(lo[1], hi[1], n)
        gx, gy = np.meshgrid(self.xs, self.ys)
        # Row-major: points[r * n + c] = (xs[c], ys[r]).
        self.points = np.column_stack([gx.ravel(), gy.ravel()])
