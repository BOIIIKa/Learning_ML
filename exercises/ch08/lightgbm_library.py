"""Gradient-boosted trees with LightGBM: assemble, train with a validation set, replay the boosting.

LightGBM grows each new tree on the gradients of the loss of the current ensemble,
so we can ask for predictions after only the first k trees and watch the boundary form.
"""

import lightgbm as lgb


def build_model(n_estimators=100, learning_rate=0.1, num_leaves=8):
    """A small LightGBM classifier (single-threaded so it also runs in the browser)."""
    # @blank model
    # zh: 创建 lgb.LGBMClassifier，传入三个超参数，并设 min_child_samples=5、n_jobs=1、verbose=-1
    # en: Create lgb.LGBMClassifier with the three hyperparameters plus min_child_samples=5, n_jobs=1, verbose=-1
    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                              min_child_samples=5, n_jobs=1, verbose=-1)
    # @end


def fit(model, X_train, y_train, X_valid, y_valid):
    """Train while tracking validation log-loss; return the per-iteration loss list."""
    # @blank fit
    # zh: 用 eval_set=[(X_valid, y_valid)]、eval_metric="binary_logloss" 训练，再从 model.evals_result_["valid_0"] 取出损失曲线
    # en: Fit with eval_set=[(X_valid, y_valid)] and eval_metric="binary_logloss", then read the curve from model.evals_result_["valid_0"]
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="binary_logloss")
    return model.evals_result_["valid_0"]["binary_logloss"]
    # @end


def staged_scores(model, points, iterations):
    """Raw ensemble scores at `points` using only the first k trees, for each k in `iterations`."""
    # @blank staged
    # zh: 对每个 k 调用 model.predict(points, raw_score=True, num_iteration=k)
    # en: For each k call model.predict(points, raw_score=True, num_iteration=k)
    return [model.predict(points, raw_score=True, num_iteration=k) for k in iterations]
    # @end
