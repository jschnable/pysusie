"""Optional numba kernels for Mr.ASH."""

from __future__ import annotations

import numpy as np


try:  # pragma: no cover - numba path depends on environment
    import numba as nb

    @nb.njit(cache=True)
    def _mrash_loop(
        X,
        w,
        sa2,
        pi,
        beta,
        r,
        sigma2,
        order,
        max_iter,
        min_iter,
        tol,
        eps,
        update_pi,
        update_sigma2,
    ):
        n, p = X.shape
        K = sa2.shape[0]
        elbo = np.empty(max_iter)
        converged = False

        for it in range(max_iter):
            pi_accum = np.zeros(K)
            max_change = 0.0

            for jj in range(p):
                j = order[jj]
                xj = X[:, j]
                wj = w[j] + eps

                old = beta[j]
                # partial residual regression coefficient
                xtr = 0.0
                for i in range(n):
                    xtr += xj[i] * (r[i] + xj[i] * old)
                b_tilde = xtr / wj

                log_post = np.empty(K)
                mu_k = np.empty(K)

                for k in range(K):
                    shrink = sa2[k] / (sa2[k] + sigma2 / wj)
                    mu = shrink * b_tilde
                    var = max((sigma2 / wj) * shrink, eps)
                    mu_k[k] = mu
                    diff = b_tilde - mu
                    log_post[k] = np.log(pi[k] + eps) - 0.5 * np.log(var) - 0.5 * diff * diff / var

                max_log = log_post[0]
                for k in range(1, K):
                    if log_post[k] > max_log:
                        max_log = log_post[k]

                norm = 0.0
                for k in range(K):
                    log_post[k] = np.exp(log_post[k] - max_log)
                    norm += log_post[k]

                new_beta = 0.0
                for k in range(K):
                    phi = log_post[k] / norm
                    pi_accum[k] += phi
                    new_beta += phi * mu_k[k]

                delta = new_beta - old
                beta[j] = new_beta
                if abs(delta) > max_change:
                    max_change = abs(delta)
                for i in range(n):
                    r[i] -= xj[i] * delta

            if update_pi:
                for k in range(K):
                    pi[k] = max(pi_accum[k] / p, eps)
                s = 0.0
                for k in range(K):
                    s += pi[k]
                for k in range(K):
                    pi[k] /= s

            if update_sigma2:
                rss = 0.0
                for i in range(n):
                    rss += r[i] * r[i]
                sigma2 = max(rss / n, eps)

            rss = 0.0
            for i in range(n):
                rss += r[i] * r[i]
            elbo[it] = -0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * rss / sigma2

            if it + 1 >= min_iter and max_change < tol:
                converged = True
                return beta, pi, sigma2, elbo[: it + 1], converged, it + 1

        return beta, pi, sigma2, elbo, converged, max_iter


except Exception:  # pragma: no cover - fallback path

    def _mrash_loop(
        X,
        w,
        sa2,
        pi,
        beta,
        r,
        sigma2,
        order,
        max_iter,
        min_iter,
        tol,
        eps,
        update_pi,
        update_sigma2,
    ):
        n, p = X.shape
        K = sa2.shape[0]
        elbo = []
        converged = False

        for it in range(max_iter):
            pi_accum = np.zeros(K)
            max_change = 0.0

            for j in order:
                xj = X[:, j]
                wj = w[j] + eps
                old = beta[j]
                b_tilde = np.dot(xj, r + xj * old) / wj

                shrink = sa2 / (sa2 + sigma2 / wj)
                mu_k = shrink * b_tilde
                var_k = np.maximum((sigma2 / wj) * shrink, eps)

                log_post = np.log(np.maximum(pi, eps)) - 0.5 * np.log(var_k) - 0.5 * (b_tilde - mu_k) ** 2 / var_k
                log_post -= np.max(log_post)
                phi = np.exp(log_post)
                phi /= np.sum(phi)

                pi_accum += phi
                new_beta = float(np.dot(phi, mu_k))

                delta = new_beta - old
                beta[j] = new_beta
                max_change = max(max_change, abs(delta))
                r -= xj * delta

            if update_pi:
                pi[:] = np.maximum(pi_accum / p, eps)
                pi[:] /= np.sum(pi)

            if update_sigma2:
                sigma2 = max(float(np.dot(r, r)) / n, eps)

            rss = float(np.dot(r, r))
            elbo.append(-0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * rss / sigma2)

            if it + 1 >= min_iter and max_change < tol:
                converged = True
                break

        return beta, pi, sigma2, np.array(elbo), converged, len(elbo)
