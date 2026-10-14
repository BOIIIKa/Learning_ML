"""ReLU networks in scikit-learn: build an MLP classifier, count its parameters, and count the
linear regions it actually uses on a grid of inputs."""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_mlp(hidden):
    """Pipeline: StandardScaler → MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
    max_iter=3000, random_state=0)."""
    # @blank mlp
    # zh: hidden 是元组，例如 (16, 16) 表示两个隐层、每层 16 个单元
    # en: hidden is a tuple, e.g. (16, 16) for two hidden layers of 16 units
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=hidden, activation="relu", max_iter=3000, random_state=0))
    # @end


def n_params(mlp):
    """Number of weights and biases of a fitted MLPClassifier."""
    # @blank params
    # zh: coefs_ 和 intercepts_ 是每层的权重矩阵与偏置向量
    # en: coefs_ and intercepts_ hold each layer's weight matrix and bias vector
    return int(sum(W.size + b.size for W, b in zip(mlp.coefs_, mlp.intercepts_)))
    # @end


def regions_used(pipe, X):
    """How many distinct on/off patterns of the hidden units occur on the inputs X (= linear regions visited)."""
    scaler, mlp = pipe[0], pipe[-1]
    H = scaler.transform(X)
    on = []
    # @blank regions
    # zh: 对每个隐层（除最后一层）：Z = H @ W + b，记录 Z > 0，H = max(0, Z)；按行拼接后数不同的行
    # en: For every hidden layer (all but the last): Z = H @ W + b, record Z > 0, H = max(0, Z); stack and count distinct rows
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        Z = H @ W + b
        on.append(Z > 0)
        H = np.maximum(0, Z)
    return len(np.unique(np.hstack(on), axis=0))
    # @end
