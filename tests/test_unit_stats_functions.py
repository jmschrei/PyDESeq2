"""Area D: module-level functions for testing and shrinkage (see tests/TARGETS.md).

``glm.wald_test``, ``distributions.nbinomFn``, ``distributions.nbinomGLM``,
``grid_search.grid_fit_shrink_beta`` and ``stats.lowess``, each against a reference
written in this module. ``tests/test_unit_stats.py`` imports the references for the
``DeseqStats`` tests:

- :func:`wald_reference`: DESeq2's Wald test (sandwich covariance with a ridge).
- :func:`apeglm_objective`, :func:`apeglm_grad_hess`, :func:`reference_map`: apeglm's
  negative log posterior (NB likelihood from ``helpers.nb_loglik``, normal prior on
  the coefficients that are not shrunk, Cauchy prior on the one that is) and its
  minimizer by trust-region Newton from several starts.
- :func:`apeglm_prior_var`: apeglm's adaptive prior variance.
- :func:`bh_adjust`, :func:`reference_lowess`, :func:`reference_independent_filtering`:
  Benjamini-Hochberg by hand, Cleveland's robust lowess and DESeq2's independent
  filtering (``pvalueAdjustment`` in DESeq2's ``results.R``).
"""

import numpy as np
import pytest
from scipy import optimize
from scipy import stats

from pydeseq2 import glm
from pydeseq2.distributions import nbinomFn
from pydeseq2.distributions import nbinomGLM
from pydeseq2.grid_search import grid_fit_shrink_beta
from pydeseq2.stats import lowess
from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS

# nbinomGLM stops L-BFGS-B at ftol=gtol=1e-8 on the objective divided by its value
# at beta = 0. Against reference_map that left absolute errors up to 3.5e-3 on the
# shrunk natural-log coefficient (6.4e-3 on any coefficient) and relative objective
# gaps up to 8.5e-6, over 820 simulated genes (2 to 5 coefficients, 6 to 18
# samples, adaptive and unit prior scales). The tolerances below leave a 3x margin
# on the coefficients and 10x on the objective.
SHRINK_ATOL = 2e-2
SHRINK_OBJECTIVE_RTOL = 1e-4

# Two starts of reference_map that end further apart than this mean the posterior
# has two local minima; such genes are left out of coefficient comparisons.
BIMODAL_SPREAD = 1e-3


# References


def wald_reference(x, disp, beta, mu, ridge, contrast, lfc_null=0.0):
    """DESeq2's Wald test for one gene, without alternative hypotheses.

    The covariance of the ridge-penalized estimate is the sandwich
    ``(M + R)^-1 M (M + R)^-1`` with ``M = X^T W X`` and ``W = mu / (1 + disp mu)``.
    Returns ``(p_value, statistic, se)``.
    """
    w = mu / (1.0 + disp * mu)
    m = (x.T * w) @ x
    hc = np.linalg.solve(m + ridge, contrast)
    se = np.sqrt(hc @ m @ hc)
    stat = (contrast @ beta - lfc_null) / se
    return 2 * stats.norm.sf(abs(stat)), stat, se


def apeglm_objective(beta, x, counts, size, offset, shrink_index, prior_scale):
    """apeglm's negative log posterior, up to a constant in ``beta``.

    NB likelihood with mean ``exp(x @ beta + offset)`` and dispersion ``1 / size``,
    a N(0, 15^2) prior on every coefficient but ``shrink_index``, and a Cauchy prior
    of scale ``prior_scale`` on that one (``log1p((b / s)^2)`` up to a constant).
    """
    beta = np.asarray(beta, dtype=float)
    mu = np.exp(x @ beta + offset)
    nll = -helpers.nb_loglik(counts, mu, 1.0 / size)
    others = np.delete(beta, shrink_index)
    return (
        nll
        + np.sum(others**2) / (2 * 15.0**2)
        + np.log1p((beta[shrink_index] / prior_scale) ** 2)
    )


def apeglm_grad_hess(beta, x, counts, size, offset, shrink_index, prior_scale):
    """Gradient and Hessian of :func:`apeglm_objective`."""
    mu = np.exp(x @ beta + offset)
    grad = x.T @ (size * (mu - counts) / (size + mu))
    hess = (x.T * (size * mu * (size + counts) / (size + mu) ** 2)) @ x
    prior_grad = beta / 15.0**2
    prior_curv = np.full(beta.size, 1 / 15.0**2)
    b, s2 = beta[shrink_index], prior_scale**2
    prior_grad[shrink_index] = 2 * b / (s2 + b**2)
    prior_curv[shrink_index] = 2 * (s2 - b**2) / (s2 + b**2) ** 2
    return grad + prior_grad, hess + np.diag(prior_curv)


def reference_map(x, counts, size, offset, shrink_index, prior_scale, starts):
    """Minimize :func:`apeglm_objective` by trust-region Newton from each start.

    Returns the lowest minimum found and the largest distance from it to the other
    starts' minima: a spread above ``BIMODAL_SPREAD`` means two local minima.
    """
    args = (x, counts, size, offset, shrink_index, prior_scale)
    fits = [
        optimize.minimize(
            apeglm_objective,
            np.asarray(start, dtype=float),
            args=args,
            jac=lambda b, *a: apeglm_grad_hess(b, *a)[0],
            hess=lambda b, *a: apeglm_grad_hess(b, *a)[1],
            method="trust-exact",
            options={"gtol": 1e-9, "maxiter": 1000},
        )
        for start in starts
    ]
    best = min(fits, key=lambda fit: fit.fun)
    spread = max(np.max(np.abs(fit.x - best.x)) for fit in fits)
    return best.x, spread


def apeglm_prior_var(mle, se, min_var=1e-6, max_var=400.0):
    """apeglm's ``priorVar``: the zero of its estimating equation, or ``min_var``.

    ``mle`` and ``se`` are the unshrunk coefficients and their standard errors
    (natural log); the equation is
    ``sum((mle^2 - se^2) w) / sum(w) = a`` with ``w = 1 / (2 (a + se^2)^2)``.
    """
    s2, d2 = np.asarray(mle) ** 2, np.asarray(se) ** 2

    def excess(a):
        w = 1 / (2 * (a + d2) ** 2)
        return np.sum((s2 - d2) * w) / np.sum(w) - a

    if excess(min_var) < 0:
        return min_var
    return optimize.brentq(excess, min_var, max_var, xtol=1e-15, rtol=1e-14)


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values, by hand (no NaN allowed)."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def reference_lowess(x, y, frac, iterations):
    """Cleveland's robust locally weighted linear regression.

    The fit at ``x[i]`` is a weighted least-squares line with tricube weights over
    the distance to ``x[i]`` divided by the ``r``-th smallest such distance,
    ``r = ceil(frac * n)`` (counting the zero distance to ``x[i]`` itself, as in
    Biopython's ``lowess``, which ``pydeseq2.stats.lowess`` follows). ``iterations``
    fits are made; each after the first multiplies the weights by the bisquare of the
    previous residuals over six times their median absolute value (weight 1 for zero
    residuals and 0 otherwise when that median is zero).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = x.size
    r = int(np.ceil(frac * n))
    dist = np.abs(x[:, None] - x[None, :])
    bandwidth = np.maximum(np.sort(dist, axis=1)[:, r], 1e-12)
    local = (1 - np.clip(dist / bandwidth[:, None], 0.0, 1.0) ** 3) ** 3
    robust = np.ones(n)
    fit = np.empty(n)
    for _ in range(iterations):
        for i in range(n):
            w = local[i] * robust
            xm, ym = np.sum(w * x) / np.sum(w), np.sum(w * y) / np.sum(w)
            sxx = np.sum(w * (x - xm) ** 2)
            slope = np.sum(w * (x - xm) * (y - ym)) / sxx if sxx > 0 else 0.0
            fit[i] = ym + slope * (x[i] - xm)
        resid = y - fit
        s = np.median(np.abs(resid))
        if s == 0:
            robust = (resid == 0).astype(float)
        else:
            robust = (1 - np.clip(resid / (6 * s), -1, 1) ** 2) ** 2
    return fit


def reference_independent_filtering(base_mean, pvalues, alpha):
    """DESeq2's independent filtering on the mean of normalized counts.

    For 50 quantiles ``theta`` of ``base_mean`` from the fraction of zeros to 0.95,
    BH-adjust the p-values of the genes at or above the quantile; count rejections at
    ``alpha``; smooth the counts with lowess (``frac=1/5``); pick the first ``theta``
    whose count exceeds the smooth's maximum minus the root mean square residual of
    the non-zero counts (the first ``theta`` when no count exceeds 10).

    Returns the adjusted p-values (NaN for filtered genes) and the chosen index.
    """
    base_mean = np.asarray(base_mean, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    lower = np.mean(base_mean == 0)
    theta = np.linspace(lower, 0.95 if lower < 0.95 else 1.0, 50)
    cutoffs = np.quantile(base_mean, theta)
    adjusted = np.full((theta.size, base_mean.size), np.nan)
    for i, cutoff in enumerate(cutoffs):
        use = (base_mean >= cutoff) & ~np.isnan(pvalues)
        if use.any():
            adjusted[i, use] = bh_adjust(pvalues[use])
    with np.errstate(invalid="ignore"):
        num_rej = np.sum(adjusted < alpha, axis=1)
    smooth = reference_lowess(theta, num_rej, frac=1 / 5, iterations=3)
    chosen = 0
    if num_rej.max() > 10:
        positive = num_rej > 0
        resid = num_rej[positive] - smooth[positive]
        thresh = smooth.max() - np.sqrt(np.mean(resid**2))
        above = np.flatnonzero(num_rej > thresh)
        chosen = int(above[0]) if above.size else 0
    return adjusted[chosen], chosen


def _two_group_design(n_per_group):
    return np.column_stack(
        [np.ones(2 * n_per_group), np.repeat([0.0, 1.0], n_per_group)]
    )


# glm.wald_test


def test_wald_test_two_groups_hand_worked():
    """Two groups of two samples, no ridge, contrast on the group coefficient.

    ``W = mu / (1 + disp mu)`` is 1 in group A (mu 2, disp 0.5) and 1.6 in group B
    (mu 8). Without a ridge the sandwich reduces to ``M^-1``, and the variance of the
    difference of the group log means is ``1 / sum(W_A) + 1 / sum(W_B) =
    1/2 + 1/3.2 = 0.8125``. The statistic is ``log(4) / sqrt(0.8125)``.
    """
    x = _two_group_design(2)
    lfc = np.array([np.log(2.0), np.log(4.0)])
    p, stat, se = glm.wald_test(
        design_matrix=x,
        disp=0.5,
        lfc=lfc,
        mu=np.array([2.0, 2.0, 8.0, 8.0]),
        ridge_factor=np.zeros((2, 2)),
        contrast=np.array([0.0, 1.0]),
        lfc_null=0.0,
        alt_hypothesis=None,
    )
    # A 2 x 2 inverse and a few products: rounding only.
    assert se == pytest.approx(np.sqrt(0.8125), rel=1e-12)
    assert stat == pytest.approx(np.log(4.0) / np.sqrt(0.8125), rel=1e-12)
    assert p == pytest.approx(
        2 * stats.norm.sf(np.log(4.0) / np.sqrt(0.8125)), rel=1e-10
    )


@pytest.mark.parametrize("design", ["two_group", "five_coefficients"])
def test_wald_test_matches_sandwich_formula(design):
    """Random genes against :func:`wald_reference`, with the default and a large ridge.

    The two computations differ only in rounding (inverse versus solve, order of
    products); 1e-9 relative on SE and statistic covers condition numbers up to 1e6.
    p-values amplify a relative error in the statistic by about ``stat^2``; they are
    compared at 1e-7.
    """
    rs = np.random.RandomState([4, 0 if design == "two_group" else 1])
    if design == "two_group":
        x = _two_group_design(5)
    else:
        sim = helpers.simulate_counts(
            n_genes=2,
            n_samples=18,
            seed=4,
            factors={"condition": ["A", "B"], "group": ["X", "Y", "Z"]},
            continuous=("age",),
        )
        x = sim.design_matrix.to_numpy()
    n, p = x.shape
    for ridge in (np.diag(np.full(p, 1e-6)), np.diag(rs.uniform(0.5, 2.0, size=p))):
        for _ in range(10):
            beta = rs.normal(size=p) + np.r_[rs.uniform(1, 8), np.zeros(p - 1)]
            mu = np.exp(x @ beta) * np.exp(rs.normal(scale=0.3, size=n))
            disp = 10 ** rs.uniform(-3, 0)
            contrast = rs.normal(size=p)
            actual = glm.wald_test(
                design_matrix=x,
                disp=disp,
                lfc=beta,
                mu=mu,
                ridge_factor=ridge,
                contrast=contrast,
                lfc_null=0.0,
                alt_hypothesis=None,
            )
            expected = wald_reference(x, disp, beta, mu, ridge, contrast)
            assert actual[2] == pytest.approx(expected[2], rel=1e-9)
            assert actual[1] == pytest.approx(expected[1], rel=1e-9)
            assert actual[0] == pytest.approx(expected[0], rel=1e-7)


# DESeq2's results() with lfcThreshold T >= 0, contrast LFC b and standard error s
# (log2 in DESeq2; the scale cancels): greaterAbs: stat sign(b) max((|b| - T)/s, 0),
# p min(1, 2 sf((|b| - T)/s)); lessAbs: stat min((T - b)/s, (b + T)/s), p
# max(cdf((b - T)/s), sf((b + T)/s)); greater: stat max((b - T)/s, 0), p
# sf((b - T)/s); less: stat min((b + T)/s, 0), p cdf((b + T)/s). PyDESeq2 passes the
# null value itself as lfc_null, so DESeq2's "less" with threshold T is lfc_null = -T.
# PyDESeq2 reports lessAbs's statistic with a sign; its magnitude is DESeq2's. Where
# DESeq2's greater, less or lessAbs statistic is clipped to zero, PyDESeq2's p-value
# differs (known-bugs/D-alt-hypothesis-clipped-p-values.md), so p-values are compared
# only where the statistic is not clipped.
@pytest.mark.parametrize(
    "alt, threshold, lfc",
    [
        ("greaterAbs", 0.5, 1.2),
        ("greaterAbs", 0.5, -1.4),
        ("greaterAbs", 0.5, 0.3),
        ("lessAbs", 0.5, 0.1),
        ("lessAbs", 0.5, -0.35),
        ("lessAbs", 0.5, 0.9),
        ("greater", 0.5, 1.3),
        ("greater", 0.5, 0.2),
        ("less", 0.5, -1.1),
        ("less", 0.5, -0.2),
    ],
)
def test_wald_test_alternative_hypotheses(alt, threshold, lfc):
    """``alt_hypothesis`` against DESeq2's formulas, on a unit contrast vector."""
    x = _two_group_design(4)
    beta = np.array([3.0, lfc])
    mu = np.exp(x @ beta)
    ridge = np.diag([1e-6, 1e-6])
    contrast = np.array([0.0, 1.0])
    _, _, s = wald_reference(x, 0.2, beta, mu, ridge, contrast)
    t, b = threshold, lfc
    if alt == "greaterAbs":
        stat = np.sign(b) * max((abs(b) - t) / s, 0.0)
        pval = min(1.0, 2 * stats.norm.sf((abs(b) - t) / s))
    elif alt == "lessAbs":
        stat = min(max((t - b) / s, 0.0), max((b + t) / s, 0.0))
        pval = max(stats.norm.cdf((b - t) / s), stats.norm.sf((b + t) / s))
    elif alt == "greater":
        stat = max((b - t) / s, 0.0)
        pval = stats.norm.sf((b - t) / s)
    else:
        stat = min((b + t) / s, 0.0)
        pval = stats.norm.cdf((b + t) / s)

    p, statistic, se = glm.wald_test(
        design_matrix=x,
        disp=0.2,
        lfc=beta,
        mu=mu,
        ridge_factor=ridge,
        contrast=contrast,
        lfc_null=-t if alt == "less" else t,
        alt_hypothesis=alt,
    )
    # Rounding only (see test_wald_test_matches_sandwich_formula).
    assert se == pytest.approx(s, rel=1e-9)
    if alt == "lessAbs":
        statistic = abs(statistic)
    assert statistic == pytest.approx(stat, rel=1e-9, abs=1e-12)
    if stat != 0 or alt == "greaterAbs":
        assert p == pytest.approx(pval, rel=1e-8)


# distributions.nbinomFn


@pytest.mark.parametrize("shrink_index", [1, 2])
def test_nbinomFn_matches_likelihood_and_prior(shrink_index):
    """Differences of nbinomFn between coefficient vectors equal those of the reference.

    nbinomFn drops the terms of the NB log-likelihood that do not depend on beta, so
    only differences are comparable. The reference is ``helpers.nb_loglik`` (accurate
    to about 1e-12 relative) plus the priors; 1e-9 relative to the objective's size
    covers the cancellation in the difference.
    """
    sim = helpers.simulate_counts(
        n_genes=3, n_samples=12, seed=6, factors={"condition": ["A", "B", "C"]}
    )
    x = sim.design_matrix.to_numpy()
    offset = np.log(sim.size_factors.to_numpy())
    rs = np.random.RandomState(6)
    for gene in sim.counts.columns:
        y = sim.counts[gene].to_numpy()
        size = 1.0 / sim.dispersions[gene]
        betas = [
            sim.beta.loc[gene].to_numpy() + rs.normal(scale=0.5, size=3)
            for _ in range(4)
        ]
        for scale in (0.3, 2.0):
            actual = [
                nbinomFn(b, x, y, size, offset, 15.0, scale, shrink_index) for b in betas
            ]
            expected = [
                apeglm_objective(b, x, y, size, offset, shrink_index, scale)
                for b in betas
            ]
            size_of_objective = max(abs(v) for v in expected)
            np.testing.assert_allclose(
                np.diff(actual), np.diff(expected), rtol=0, atol=1e-9 * size_of_objective
            )


# distributions.nbinomGLM


def _map_cases():
    """Simulated genes and their designs for the MAP comparisons."""
    two = helpers.simulate_counts(n_genes=25, n_samples=8, seed=7, disp_scatter=0.3)
    multi = helpers.simulate_counts(
        n_genes=20,
        n_samples=12,
        seed=8,
        disp_scatter=0.3,
        factors={"condition": ["A", "B"], "group": ["X", "Y"]},
        continuous=("age",),
    )
    return [(two, 1), (multi, 2), (multi, 3)]


@pytest.mark.parametrize("prior_scale", [0.2, 1.0])
def test_nbinomGLM_matches_reference_map(prior_scale):
    """MAP coefficients against :func:`reference_map`, gene by gene.

    Uses the true dispersions and size factors, so the inputs are what nbinomGLM
    documents and nothing else is under test. ``converged`` must be True; genes with
    two local minima are left out (at most 2 of 65 per prior scale).
    """
    n_bimodal = 0
    for sim, shrink_index in _map_cases():
        x = sim.design_matrix.to_numpy()
        offset = np.log(sim.size_factors.to_numpy())
        for gene in sim.counts.columns:
            y = sim.counts[gene].to_numpy()
            size = 1.0 / sim.dispersions[gene]
            beta, _, converged = nbinomGLM(
                x, y, size, offset, 15.0, prior_scale, shrink_index=shrink_index
            )
            ref, spread = reference_map(
                x,
                y,
                size,
                offset,
                shrink_index,
                prior_scale,
                [sim.beta.loc[gene].to_numpy(), np.zeros(x.shape[1])],
            )
            assert converged, gene
            if spread > BIMODAL_SPREAD:
                n_bimodal += 1
                continue
            args = (x, y, size, offset, shrink_index, prior_scale)
            f_ref = apeglm_objective(ref, *args)
            gap = (apeglm_objective(beta, *args) - f_ref) / abs(f_ref)
            assert gap <= SHRINK_OBJECTIVE_RTOL, (gene, shrink_index, gap)
            np.testing.assert_allclose(beta, ref, rtol=0, atol=SHRINK_ATOL, err_msg=gene)
    assert n_bimodal <= 2


def test_nbinomGLM_inverse_hessian_under_a_weak_prior():
    """``inv_hessian`` against the inverse of the reference Hessian at the MAP.

    With prior scale 10 the prior's curvature (at most 2 / 10^2 = 0.02 on the shrunk
    coefficient, 1/225 on the others) is below 1e-3 of the smallest diagonal entry of
    the likelihood's curvature on these genes (checked below), so how the prior terms
    enter the Hessian moves its inverse by less than 1e-3 (see
    known-bugs/D-shrink-hessian-prior-broadcast.md for the case of a strong prior).
    The MAP is within SHRINK_ATOL of the reference, which moves the curvature by
    under 2%; 5e-2 relative on the diagonal of the inverse covers both.
    """
    sim = helpers.simulate_counts(
        n_genes=12, n_samples=10, seed=9, mean_range=(100, 5000), disp_scatter=0.3
    )
    x = sim.design_matrix.to_numpy()
    offset = np.log(sim.size_factors.to_numpy())
    for gene in sim.counts.columns:
        y = sim.counts[gene].to_numpy()
        size = 1.0 / sim.dispersions[gene]
        beta, inv_hessian, converged = nbinomGLM(x, y, size, offset, 15.0, 10.0)
        assert converged
        _, hess = apeglm_grad_hess(beta, x, y, size, offset, 1, 10.0)
        likelihood_curv = hess - np.diag([1 / 225, 0.0])
        likelihood_curv[1, 1] -= 2 * (100 - beta[1] ** 2) / (100 + beta[1] ** 2) ** 2
        assert 0.02 < 1e-3 * np.diag(likelihood_curv).min()
        np.testing.assert_allclose(
            np.diag(inv_hessian), np.diag(np.linalg.inv(hess)), rtol=5e-2, err_msg=gene
        )


def test_nbinomGLM_genes_with_zero_group():
    """Genes with all-zero counts in one group: the prior keeps the MAP finite.

    The MLE of the group coefficient is at minus or plus infinity; the posterior's
    minimum is where the Cauchy prior's pull balances a likelihood that flattens
    out, so it is poorly conditioned along that coefficient. Compared through the
    objective gap (SHRINK_OBJECTIVE_RTOL) and the sign and size of the coefficient
    (5% relative: the largest difference measured was 6e-3). The low-count gene has
    two local minima under the narrow prior and is compared only under the wide one.
    """
    x = _two_group_design(4)
    offset = np.zeros(8)
    n_compared = 0
    for counts, size in [
        ([40, 55, 61, 38, 0, 0, 0, 0], 5.0),
        ([0, 0, 0, 0, 900, 1200, 750, 1010], 20.0),
        ([0, 0, 0, 0, 3, 1, 4, 2], 2.0),
    ]:
        y = np.asarray(counts, dtype=float)
        for scale in (0.1, 1.0):
            beta, _, converged = nbinomGLM(x, y, size, offset, 15.0, scale)
            far = np.array([0.0, 5.0 * np.sign(y[4] - y[0])])
            ref, spread = reference_map(x, y, size, offset, 1, scale, [np.zeros(2), far])
            assert converged
            if spread > BIMODAL_SPREAD:
                continue
            n_compared += 1
            args = (x, y, size, offset, 1, scale)
            f_ref = apeglm_objective(ref, *args)
            assert (apeglm_objective(beta, *args) - f_ref) / abs(
                f_ref
            ) <= SHRINK_OBJECTIVE_RTOL
            assert np.sign(beta[1]) == np.sign(ref[1])
            assert beta[1] == pytest.approx(ref[1], rel=0.05)
    assert n_compared == 5


# grid_search.grid_fit_shrink_beta


def test_grid_fit_shrink_beta_matches_reference():
    """The 2D grid minimum against :func:`reference_map`.

    The coarse grid has 60 points over [-30, 30] (step 60/59); the fine grid has 60
    points over two coarse steps around the coarse minimum (step 2 (60/59) / 59, about
    0.0345). On these well-conditioned genes the fine-grid minimum is within two fine
    steps of the continuous minimum in each coordinate, and its objective is within
    the objective's change over one fine step.
    """
    fine_step = 2 * (60 / 59) / 59
    sim = helpers.simulate_counts(n_genes=4, n_samples=8, seed=10, frac_de=1.0)
    x = sim.design_matrix.to_numpy()
    offset = np.log(sim.size_factors.to_numpy())
    for gene in sim.counts.columns:
        y = sim.counts[gene].to_numpy()
        size = 1.0 / sim.dispersions[gene]
        scale_cnst = max(nbinomFn(np.zeros(2), x, y, size, offset, 15.0, 0.5), 1.0)
        beta = grid_fit_shrink_beta(y, offset, x, size, 15.0, 0.5, scale_cnst)
        ref, spread = reference_map(
            x, y, size, offset, 1, 0.5, [sim.beta.loc[gene].to_numpy(), np.zeros(2)]
        )
        assert spread < BIMODAL_SPREAD
        np.testing.assert_allclose(beta, ref, rtol=0, atol=2 * fine_step, err_msg=gene)
        _, hess = apeglm_grad_hess(ref, x, y, size, offset, 1, 0.5)
        args = (x, y, size, offset, 1, 0.5)
        step = np.full(2, fine_step)
        assert (
            apeglm_objective(beta, *args) - apeglm_objective(ref, *args)
            <= step @ hess @ step
        )


# stats.lowess


def test_lowess_reproduces_a_line():
    """One local linear fit (``iter=1``) of exactly linear data is the line.

    Only ``iter=1``: with robustness iterations the residuals of an exact fit are at
    the rounding level and their bisquare weights are arbitrary
    (known-bugs/D-lowess-exact-fit.md).
    """
    x = np.linspace(0.1, 0.95, 50)
    y = 3.0 - 7.5 * x
    for frac in (0.1, 1 / 5, 2 / 3):
        # Weighted least squares on exact data: rounding only.
        np.testing.assert_allclose(lowess(x, y, frac=frac, iter=1), y, rtol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lowess_matches_reference(seed):
    """Noisy curves with a few gross outliers against :func:`reference_lowess`.

    The two solve the same weighted least-squares problems in different
    arrangements (normal equations versus centred sums), which differ by rounding;
    1e-8 relative to the range of y covers the robustness iterations' amplification.
    """
    rs = np.random.RandomState([11, seed])
    x = np.sort(rs.uniform(0, 1, size=50))
    y = 40 * np.sin(3 * x) + rs.normal(scale=2.0, size=50)
    y[rs.choice(50, size=3, replace=False)] += 60.0
    for frac, iterations in [(1 / 5, 3), (0.5, 2)]:
        np.testing.assert_allclose(
            lowess(x, y, frac=frac, iter=iterations),
            reference_lowess(x, y, frac, iterations),
            rtol=0,
            atol=1e-8 * np.ptp(y),
        )


def test_reference_independent_filtering_hand_worked():
    """Self-test of :func:`reference_independent_filtering` on a small worked case.

    With no more than 10 rejections at any quantile the first quantile is chosen,
    which keeps every gene with a non-zero mean; the adjusted p-values are then BH
    over those genes (worked below).
    """
    base_mean = np.array([0.0, 5.0, 10.0, 20.0, 40.0])
    p = np.array([np.nan, 0.01, 0.04, 0.03, 0.5])
    adjusted, chosen = reference_independent_filtering(base_mean, p, alpha=0.05)
    # Sorted p: 0.01, 0.03, 0.04, 0.5 -> scaled 0.04, 0.06, 0.0533, 0.5 -> running
    # minimum from the top: 0.04, 0.0533, 0.0533, 0.5.
    assert chosen == 0
    np.testing.assert_allclose(
        adjusted, [np.nan, 0.04, 0.16 / 3, 0.16 / 3, 0.5], rtol=1e-12
    )
