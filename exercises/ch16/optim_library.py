"""Optimizers in SciPy and scikit-learn: quasi-Newton and Newton methods on Rosenbrock, and an MLP
trained with plain SGD, Nesterov momentum and Adam."""

import warnings

from scipy.optimize import minimize, rosen, rosen_der, rosen_hess
from sklearn.neural_network import MLPClassifier

METHODS = ["CG", "BFGS", "L-BFGS-B", "Newton-CG"]


def scipy_race(x0):
    """Minimise Rosenbrock from x0 with each method in METHODS, giving it the gradient (and, for
    Newton-CG, the Hessian). Returns {method: OptimizeResult}."""
    out = {}
    for m in METHODS:
        # @blank minimize
        # zh: minimize(rosen, x0, method=m, jac=rosen_der)；Newton-CG 还要 hess=rosen_hess
        # en: minimize(rosen, x0, method=m, jac=rosen_der); Newton-CG also takes hess=rosen_hess
        out[m] = minimize(rosen, x0, method=m, jac=rosen_der, hess=rosen_hess if m == "Newton-CG" else None)
        # @end
    return out


def make_mlp(kind):
    """MLPClassifier((16, 16)) with batch_size=16, max_iter=120, random_state=0, and
    "sgd": solver="sgd", learning_rate_init=0.02, momentum=0;
    "nesterov": the same with momentum=0.9, nesterovs_momentum=True;
    "adam": solver="adam", learning_rate_init=0.01."""
    common = dict(hidden_layer_sizes=(16, 16), batch_size=16, max_iter=120, random_state=0, n_iter_no_change=1000)
    # @blank mlp
    # zh: 三种 kind 只差在 solver、学习率和动量参数
    # en: The three kinds differ only in solver, learning rate and the momentum settings
    if kind == "sgd":
        return MLPClassifier(solver="sgd", learning_rate_init=0.02, momentum=0, **common)
    if kind == "nesterov":
        return MLPClassifier(solver="sgd", learning_rate_init=0.02, momentum=0.9, nesterovs_momentum=True, **common)
    return MLPClassifier(solver="adam", learning_rate_init=0.01, **common)
    # @end


def loss_curves(X, y):
    """Fit the three MLPs; return {kind: list of training losses per epoch} (the loss_curve_ attribute)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # @blank curves
        # zh: 拟合后读出 loss_curve_
        # en: Fit, then read loss_curve_
        return {k: make_mlp(k).fit(X, y).loss_curve_ for k in ["sgd", "nesterov", "adam"]}
        # @end
