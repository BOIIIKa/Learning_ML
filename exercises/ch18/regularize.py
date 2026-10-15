"""Explicit regularisers from scratch: inverted dropout, decoupled weight decay and early stopping,
for a ReLU network trained with minibatch SGD."""

import numpy as np


def dropout(h, p, rng, train=True):
    """Inverted dropout. In training, zero each entry with probability p and divide the survivors by
    1 − p, so the expectation is unchanged; return (output, mask). At test time return (h, None)."""
    if not train or p == 0:
        return h, None
    # @blank dropout
    # zh: mask = (rng.uniform(size=h.shape) >= p) / (1 − p)，输出 h * mask
    # en: mask = (rng.uniform(size=h.shape) >= p) / (1 − p); the output is h * mask
    mask = (rng.uniform(size=h.shape) >= p) / (1 - p)
    return h * mask, mask
    # @end


def sgd_step(params, grads, lr, wd):
    """One SGD step with decoupled weight decay on the weights (not the biases): W ← (1 − lr·wd) W − lr·dW."""
    out = []
    for (W, b), (dW, db) in zip(params, grads):
        # @blank decay
        # zh: 权重先乘 (1 − lr·wd) 再减去 lr·dW；偏置只做普通的梯度步
        # en: Multiply W by (1 − lr·wd), then subtract lr·dW; the bias takes a plain gradient step
        out.append(((1 - lr * wd) * W - lr * dW, b - lr * db))
        # @end
    return out


def forward(params, X, p, rng, train=True, masks=None):
    """ReLU MLP with dropout after every hidden layer; linear output. Pass `masks` to reuse fixed masks.
    Returns (f, cache) with f the logits and cache = (hs, zs, masks)."""
    hs, zs, used = [X], [], []
    for l, (W, b) in enumerate(params):
        z = hs[-1] @ W.T + b
        zs.append(z)
        if l == len(params) - 1:
            hs.append(z)
            break
        h = np.maximum(z, 0)
        if masks is not None:
            m = masks[l]
            h = h * m if m is not None else h
        else:
            h, m = dropout(h, p, rng, train)
        used.append(m)
        hs.append(h)
    return hs[-1][:, 0], (hs, zs, used)


def loss(params, X, y, p=0.0, rng=None, train=False, masks=None):
    """Mean binary cross-entropy with logits, labels y ∈ {0, 1}."""
    f, _ = forward(params, X, p, rng, train, masks)
    return float(np.mean(np.logaddexp(0, f) - y * f))


def backward(params, X, y, cache):
    """Gradients of the mean BCE through the cached forward pass (dropout masks included)."""
    hs, zs, masks = cache
    delta = ((1 / (1 + np.exp(-hs[-1][:, 0])) - y) / len(y))[:, None]
    grads = [None] * len(params)
    for l in range(len(params) - 1, -1, -1):
        W, _ = params[l]
        grads[l] = (delta.T @ hs[l], delta.sum(axis=0))
        if l > 0:
            back = delta @ W
            # @blank mask
            # zh: 梯度也要乘同一个掩码（若有），再乘 ReLU 的导数 [z > 0]
            # en: The gradient is multiplied by the same mask (if any), then by ReLU's derivative [z > 0]
            if masks[l - 1] is not None:
                back = back * masks[l - 1]
            delta = back * (zs[l - 1] > 0)
            # @end
    return grads


def early_stopping(step, val_loss, max_epochs, patience):
    """Call step() once per epoch (it returns the current params) and val_loss(params) after it.
    Stop when the validation loss has not improved for `patience` epochs.
    Return (best_params, best_epoch, epochs_run), epochs counted from 1."""
    best, best_epoch, best_params = np.inf, 0, None
    for epoch in range(1, max_epochs + 1):
        params = step()
        v = val_loss(params)
        # @blank early
        # zh: 更好就记下（参数要复制）；否则若已经 patience 轮没有进步就返回
        # en: Record an improvement (copy the params); otherwise return once `patience` epochs passed without one
        if v < best:
            best, best_epoch, best_params = v, epoch, [(W.copy(), b.copy()) for W, b in params]
        elif epoch - best_epoch >= patience:
            return best_params, best_epoch, epoch
        # @end
    return best_params, best_epoch, max_epochs


def init(sizes, rng):
    return [(rng.normal(0, np.sqrt(2 / a), (b, a)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]


def train(X, y, Xv, yv, sizes, p=0.0, wd=0.0, lr=0.1, batch=16, max_epochs=300, patience=None, seed=0, trace=None):
    """Minibatch SGD with dropout p and weight decay wd; early stopping on (Xv, yv) if `patience` is set."""
    rng = np.random.default_rng(seed)
    state = {"params": init(sizes, rng), "epoch": 0}

    def step():
        params = state["params"]
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            i = order[s:s + batch]
            _, cache = forward(params, X[i], p, rng, True)
            params = sgd_step(params, backward(params, X[i], y[i], cache), lr, wd)
        state["params"], state["epoch"] = params, state["epoch"] + 1
        if trace is not None:
            trace(state["epoch"], params)
        return params

    if patience is None:
        for _ in range(max_epochs):
            step()
        return state["params"], max_epochs, max_epochs
    return early_stopping(step, lambda q: loss(q, Xv, yv), max_epochs, patience)
