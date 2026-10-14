"""Hidden Markov models from scratch: forward, Viterbi and one Baum–Welch (EM) step.

pi: (N,), A: (N, N) with A[i, j] = P(s_{t+1} = j | s_t = i), B: (N, M) with B[j, o] = P(o | s = j),
O: integer observations of length T.
"""

import numpy as np


def forward(pi, A, B, O):
    """Scaled forward pass. Returns (alpha_hat (T, N) with rows summing to 1, log P(O))."""
    T = len(O)
    al = np.zeros((T, len(pi)))
    logp = 0.0
    for t in range(T):
        # @blank forward
        # zh: t = 0 时 a = π ⊙ B[:, o_0]；否则 a = (α̂_{t−1} A) ⊙ B[:, o_t]。再除以 c = a.sum() 并累加 log c
        # en: At t = 0, a = π ⊙ B[:, o_0]; otherwise a = (α̂_{t−1} A) ⊙ B[:, o_t]. Then divide by c = a.sum() and add log c
        a = pi * B[:, O[0]] if t == 0 else (al[t - 1] @ A) * B[:, O[t]]
        c = a.sum()
        al[t] = a / c
        logp += np.log(c)
        # @end
    return al, logp


def viterbi(pi, A, B, O):
    """Most likely state sequence, computed in log space. Returns (path, log probability)."""
    T, N = len(O), len(pi)
    lA, lB = np.log(A), np.log(B)
    d = np.log(pi) + lB[:, O[0]]
    back = np.zeros((T, N), dtype=int)
    for t in range(1, T):
        # @blank viterbi
        # zh: cand[i, j] = d[i] + log A[i, j]；back[t] 取每列的 argmax，新 d = 每列最大值 + log B[:, o_t]
        # en: cand[i, j] = d[i] + log A[i, j]; back[t] is the argmax of each column, the new d is the column max + log B[:, o_t]
        cand = d[:, None] + lA
        back[t] = cand.argmax(axis=0)
        d = cand.max(axis=0) + lB[:, O[t]]
        # @end
    path = [int(d.argmax())]
    for t in range(T - 1, 0, -1):
        # @blank backtrack
        # zh: 沿回溯指针往回走：前一个状态 = back[t][当前状态]
        # en: Follow the back-pointers: the previous state is back[t][current state]
        path.append(int(back[t][path[-1]]))
        # @end
    return path[::-1], float(d.max())


def backward(A, B, O, c):
    """Scaled backward pass using the forward scaling factors c (length T)."""
    T, N = len(O), A.shape[0]
    be = np.ones((T, N))
    for t in range(T - 2, -1, -1):
        be[t] = (A @ (B[:, O[t + 1]] * be[t + 1])) / c[t + 1]
    return be


def baum_welch_step(pi, A, B, O):
    """One EM step. Returns new (pi, A, B) and the log-likelihood of the old parameters."""
    T, N = len(O), len(pi)
    al = np.zeros((T, N))
    c = np.zeros(T)
    for t in range(T):
        a = pi * B[:, O[0]] if t == 0 else (al[t - 1] @ A) * B[:, O[t]]
        c[t] = a.sum()
        al[t] = a / c[t]
    be = backward(A, B, O, c)
    gamma = al * be
    xi = np.zeros((N, N))
    for t in range(T - 1):
        xi += al[t][:, None] * A * (B[:, O[t + 1]] * be[t + 1])[None, :] / c[t + 1]
    # @blank mstep
    # zh: π̂ = γ_1；Â_ij = Σξ_ij / Σ_{t<T} γ_t(i)；B̂_j(o) = Σ_{t: o_t = o} γ_t(j) / Σ_t γ_t(j)
    # en: π̂ = γ_1; Â_ij = Σξ_ij / Σ_{t<T} γ_t(i); B̂_j(o) = Σ_{t: o_t = o} γ_t(j) / Σ_t γ_t(j)
    new_pi = gamma[0]
    new_A = xi / gamma[:-1].sum(axis=0)[:, None]
    M = B.shape[1]
    new_B = np.stack([gamma[np.asarray(O) == o].sum(axis=0) for o in range(M)], axis=1) / gamma.sum(axis=0)[:, None]
    # @end
    return new_pi, new_A, new_B, float(np.log(c).sum())
