"""Same model, three strategies: least squares, Huber and least absolute deviation (median regression)."""

from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor


def fit_three(X, y):
    """Return the fitted (LinearRegression, HuberRegressor, QuantileRegressor at the median)."""
    # @blank models
    # zh: 分别拟合 LinearRegression()、HuberRegressor()、QuantileRegressor(quantile=0.5, alpha=0)（后者是中位数回归 = 绝对值损失）
    # en: Fit LinearRegression(), HuberRegressor() and QuantileRegressor(quantile=0.5, alpha=0) (the last is median regression = absolute loss)
    ols = LinearRegression().fit(X, y)
    huber = HuberRegressor().fit(X, y)
    lad = QuantileRegressor(quantile=0.5, alpha=0).fit(X, y)
    # @end
    return ols, huber, lad
