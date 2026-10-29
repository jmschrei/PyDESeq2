"""Shared helpers for the unit tests in ``tests/test_unit_*.py``.

The rules these helpers support (independent expected values, tolerances instead of
bitwise equality, no dependence on ``n_cpus`` or test order) are written in
``tests/README_NEW_TESTS.md``.

Contents:

- :func:`get_test_cpus`, :func:`make_dds`: run PyDESeq2 at a fixed ``n_cpus``.
- :func:`simulate_counts`: seeded negative binomial counts with recorded truth.
- :func:`size_factor_example`, :func:`nb_example`: small hand-worked inputs.
- :func:`nb_logpmf`, :func:`nb_loglik`: reference likelihood (``scipy.stats.nbinom``
  in a form that stays accurate for small dispersions).
- :func:`dispersion_objective`, :func:`reference_dispersion_mle`: the Cox-Reid
  adjusted dispersion objective and its minimizer by grid plus bounded Brent search.
- :func:`glm_objective`, :func:`reference_glm_mle`: the ridge-penalized NB GLM
  objective and its minimizer by Newton's method.
- :func:`compare_fields`, :func:`assert_fields_close`: tolerance comparison that
  names the genes and fields that differ.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from formulaic import model_matrix
from scipy import optimize
from scipy import special

from pydeseq2.dds import DeseqDataSet

CPUS_ENV_VAR = "PYDESEQ2_TEST_CPUS"

# Every test_unit_*.py module sets ``pytestmark = PANDAS_DEPRECATION_FILTERS``.
# pandas 3.1.0 adds two Pandas4Warnings (DeprecationWarning subclasses) that the
# repository's ``filterwarnings = error`` turns into failures unrelated to the
# numbers under test: anndata 0.13.4 reads the deprecated
# ``pd.options.future.infer_string`` whenever an AnnData is built, and
# ``DeseqDataSet._fit_parametric_dispersion_trend`` calls ``Series.drop`` with the
# deprecated ``inplace=True``. Matching on the message (with the builtin category)
# keeps the filters valid on pandas 2.x, where Pandas4Warning does not exist.
PANDAS_DEPRECATION_FILTERS = [
    pytest.mark.filterwarnings(
        "ignore:The 'future.infer_string' option is deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:The inplace keyword in Series.drop is deprecated:DeprecationWarning"
    ),
]


def get_test_cpus() -> int:
    """Return the ``n_cpus`` the tests pass to PyDESeq2.

    Read from the ``PYDESEQ2_TEST_CPUS`` environment variable, default 1. PyDESeq2's
    own default (``n_cpus=None``) starts one worker per core, which is every core of
    the machine.
    """
    value = os.environ.get(CPUS_ENV_VAR, "1")
    n_cpus = int(value)
    if n_cpus < 1:
        raise ValueError(f"{CPUS_ENV_VAR} must be a positive integer, got {value!r}.")
    return n_cpus


def make_dds(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    design: str | pd.DataFrame = "~condition",
    **kwargs,
) -> DeseqDataSet:
    """Build a quiet ``DeseqDataSet`` at the test ``n_cpus``.

    Keyword arguments are passed to ``DeseqDataSet``; ``quiet`` defaults to True and
    ``n_cpus`` to :func:`get_test_cpus`.
    """
    kwargs.setdefault("quiet", True)
    if kwargs.get("inference") is None:
        kwargs.setdefault("n_cpus", get_test_cpus())
    return DeseqDataSet(counts=counts, metadata=metadata, design=design, **kwargs)


# Simulation


@dataclass
class SimulatedCounts:
    """Simulated counts and the truth they were drawn from.

    Gene-level truth is NaN for ``all_zero`` and ``single_sample`` genes, which are
    not drawn from a negative binomial.

    Attributes
    ----------
    counts
        Samples x genes, int64.
    metadata
        Samples x variables (string factors, float covariates).
    design
        Formula the truth was built with.
    design_matrix
        Samples x coefficients, built by formulaic from ``design``; column names are
        the ones ``DeseqDataSet.varm["LFC"]`` uses.
    beta
        Genes x coefficients, natural log scale, as in ``DeseqDataSet.varm["LFC"]``.
    dispersions
        True dispersion per gene (variance is ``mu + alpha * mu**2``).
    size_factors
        True size factors, geometric mean 1.
    mu
        Samples x genes, true means ``size_factor * exp(design_matrix @ beta)``.
    normed_means
        True mean of normalized counts, ``mean_j exp(x_j @ beta)``; the covariate of
        the dispersion trend.
    gene_kind
        One of ``"null"``, ``"de"``, ``"large"``, ``"all_zero"``, ``"single_sample"``.
    outliers
        Samples x genes, True where a count was replaced by an outlier.
    trend
        ``(a0, a1)`` of the trend ``a0 + a1 / normed_mean``.
    """

    counts: pd.DataFrame
    metadata: pd.DataFrame
    design: str
    design_matrix: pd.DataFrame
    beta: pd.DataFrame
    dispersions: pd.Series
    size_factors: pd.Series
    mu: pd.DataFrame
    normed_means: pd.Series
    gene_kind: pd.Series
    outliers: pd.DataFrame
    trend: tuple[float, float]

    @property
    def log2_fold_changes(self) -> pd.DataFrame:
        """True coefficients in log2 scale, as in ``DeseqStats.results_df``."""
        return self.beta / np.log(2)


def _stream(seed: int, component: int) -> np.random.RandomState:
    # One legacy RandomState per component: 0 covariates, 1 size factors, 2-4 regular
    # genes, 5 outliers, 6 single-sample genes, 7-9 large genes. RandomState streams
    # are frozen by NumPy's compatibility policy (Generator streams are not), so a
    # seed gives the same counts on every NumPy version, and the special-gene options
    # leave the regular genes' counts unchanged.
    return np.random.RandomState([seed, component])


def _draw_nb_genes(seed, first_stream, n, log_mean_range, x, size_factors, kw):
    # Genes drawn from streams first_stream .. first_stream + 2: coefficients in
    # natural log (column 0 is the intercept, log-uniform over log_mean_range),
    # dispersions on the trend with log-normal scatter, and NB counts as a
    # gamma-Poisson mixture (mean mu, variance mu + alpha mu^2), gene by gene.
    rs = _stream(seed, first_stream)
    intercept = rs.uniform(*log_mean_range, size=n)
    is_de = rs.uniform(size=n) < kw["frac_de"]
    effects = rs.normal(scale=kw["lfc_sd"], size=(n, x.shape[1] - 1)) * np.log(2)
    effects[~is_de] = 0.0
    beta = np.column_stack([intercept, effects])
    rel_mu = np.exp(x @ beta.T)  # samples x genes, normalized scale
    normed_means = rel_mu.mean(axis=0)
    log_scatter = _stream(seed, first_stream + 1).normal(
        scale=kw["disp_scatter"], size=n
    )
    a0, a1 = kw["trend"]
    dispersions = (a0 + a1 / normed_means) * np.exp(log_scatter)
    mu = size_factors[:, None] * rel_mu
    counts = _stream(seed, first_stream + 2).negative_binomial(
        n=1.0 / dispersions[:, None], p=1.0 / (1.0 + dispersions[:, None] * mu.T)
    )
    return {
        "beta": beta,
        "normed_means": normed_means,
        "dispersions": dispersions,
        "mu": mu,
        "counts": counts.T.astype(np.int64),
        "is_de": is_de,
    }


def simulate_counts(
    n_genes: int = 200,
    n_samples: int = 12,
    *,
    seed: int = 0,
    factors: dict[str, list[str]] | None = None,
    continuous: tuple[str, ...] = (),
    design: str | None = None,
    mean_range: tuple[float, float] = (10.0, 5000.0),
    frac_de: float = 0.3,
    lfc_sd: float = 1.0,
    trend: tuple[float, float] = (0.1, 4.0),
    disp_scatter: float = 0.0,
    size_factor_sd: float = 0.3,
    n_all_zero: int = 0,
    n_single_sample: int = 0,
    n_large: int = 0,
    large_mean: float = 1e7,
    n_outliers: int = 0,
    outlier_factor: float = 100.0,
) -> SimulatedCounts:
    """Draw negative binomial counts with known coefficients and dispersions.

    Samples are assigned to factor-level combinations in rotation, so every
    combination gets ``n_samples // n_combinations`` samples or one more. Regular
    genes come first (``gene1`` ...), then the special genes (``large1`` ...,
    ``all_zero1`` ..., ``single_sample1`` ...).

    Parameters
    ----------
    n_genes
        Number of regular genes (special genes are added on top).
    n_samples
        Number of samples.
    seed
        Seed; the same arguments and seed give the same counts on every platform.
    factors
        Categorical variables and their levels (default ``{"condition": ["A", "B"]}``).
        The first level in sorted order is formulaic's reference level.
    continuous
        Names of continuous covariates, each drawn from N(0, 1) per sample.
    design
        Formula for the truth (default: every factor and covariate, additive).
    mean_range
        Range of the per-gene mean at the reference level, drawn log-uniformly.
    frac_de
        Fraction of regular genes with non-zero non-intercept coefficients.
    lfc_sd
        Standard deviation (log2 scale) of non-zero non-intercept coefficients.
    trend
        ``(a0, a1)``: the true dispersion is ``a0 + a1 / normed_mean`` times the
        scatter. The default is DESeq2's ``makeExampleDESeqDataSet`` relation.
    disp_scatter
        Standard deviation of the log-normal scatter around the trend.
    size_factor_sd
        Standard deviation of the log size factors.
    n_all_zero
        Genes whose counts are all zero.
    n_single_sample
        Genes with a positive count in exactly one sample.
    n_large
        Regular NB genes with reference-level mean ``large_mean``.
    large_mean
        Reference-level mean of the ``large`` genes.
    n_outliers
        Number of regular genes that get one outlier count each, in a random sample:
        the count becomes ``ceil(outlier_factor * max(mu, 1))``.
    outlier_factor
        Multiplier for outlier counts.

    Returns
    -------
    The counts, metadata and truth.
    """
    if factors is None:
        factors = {"condition": ["A", "B"]}
    sample_names = [f"sample{i + 1}" for i in range(n_samples)]

    # Design: factor combinations in rotation, continuous covariates from N(0, 1).
    levels = [list(v) for v in factors.values()]
    combos = list(pd.MultiIndex.from_product(levels)) if levels else [()]
    metadata = pd.DataFrame(index=sample_names)
    for k, name in enumerate(factors):
        metadata[name] = [combos[i % len(combos)][k] for i in range(n_samples)]
    rs = _stream(seed, 0)
    for name in continuous:
        metadata[name] = rs.normal(size=n_samples)
    if design is None:
        terms = [*factors, *continuous]
        design = "~" + " + ".join(terms) if terms else "~1"
    dm = model_matrix(design, metadata)
    design_matrix = pd.DataFrame(
        np.asarray(dm, dtype=float), index=metadata.index, columns=list(dm.columns)
    )
    n_coefs = design_matrix.shape[1]

    # Size factors, geometric mean 1.
    log_sf = _stream(seed, 1).normal(scale=size_factor_sd, size=n_samples)
    size_factors = np.exp(log_sf - log_sf.mean())

    # Regular genes (streams 2-4), then large genes (streams 7-9), so turning on the
    # large genes leaves the regular genes' draws unchanged.
    x = design_matrix.to_numpy()
    kw = {
        "frac_de": frac_de,
        "lfc_sd": lfc_sd,
        "trend": trend,
        "disp_scatter": disp_scatter,
    }
    regular = _draw_nb_genes(seed, 2, n_genes, np.log(mean_range), x, size_factors, kw)
    log_large = (np.log(large_mean), np.log(large_mean))
    large = _draw_nb_genes(seed, 7, n_large, log_large, x, size_factors, kw)
    beta = np.vstack([regular["beta"], large["beta"]])
    normed_means = np.concatenate([regular["normed_means"], large["normed_means"]])
    dispersions = np.concatenate([regular["dispersions"], large["dispersions"]])
    mu = np.hstack([regular["mu"], large["mu"]])
    counts = np.hstack([regular["counts"], large["counts"]])
    is_de = np.concatenate([regular["is_de"], large["is_de"]])

    outliers = np.zeros_like(counts, dtype=bool)
    if n_outliers > 0:
        if n_outliers > n_genes:
            raise ValueError("n_outliers cannot exceed n_genes.")
        rs = _stream(seed, 5)
        genes = rs.choice(n_genes, size=n_outliers, replace=False)
        samples = rs.randint(0, n_samples, size=n_outliers)
        outliers[samples, genes] = True
        counts[samples, genes] = np.ceil(
            outlier_factor * np.maximum(mu[samples, genes], 1.0)
        ).astype(np.int64)

    kind = np.where(is_de, "de", "null").astype(object)
    kind[n_genes:] = "large"
    gene_names = [f"gene{i + 1}" for i in range(n_genes)] + [
        f"large{i + 1}" for i in range(n_large)
    ]

    # Special genes: all zero, and positive in exactly one sample.
    special_counts = np.zeros((n_samples, n_all_zero + n_single_sample), dtype=np.int64)
    if n_single_sample > 0:
        rs = _stream(seed, 6)
        for j in range(n_single_sample):
            special_counts[j % n_samples, n_all_zero + j] = rs.randint(1, 100)
    gene_names += [f"all_zero{i + 1}" for i in range(n_all_zero)]
    gene_names += [f"single_sample{i + 1}" for i in range(n_single_sample)]
    n_special = n_all_zero + n_single_sample
    nan_genes = np.full(n_special, np.nan)

    counts = np.hstack([counts, special_counts])
    outliers = np.hstack([outliers, np.zeros((n_samples, n_special), dtype=bool)])
    beta = np.vstack([beta, np.full((n_special, n_coefs), np.nan)])
    mu = np.hstack([mu, np.full((n_samples, n_special), np.nan)])
    kind = np.concatenate(
        [kind, ["all_zero"] * n_all_zero, ["single_sample"] * n_single_sample]
    )

    return SimulatedCounts(
        counts=pd.DataFrame(counts, index=sample_names, columns=gene_names),
        metadata=metadata,
        design=design,
        design_matrix=design_matrix,
        beta=pd.DataFrame(beta, index=gene_names, columns=design_matrix.columns),
        dispersions=pd.Series(
            np.concatenate([dispersions, nan_genes]), index=gene_names
        ),
        size_factors=pd.Series(size_factors, index=sample_names),
        mu=pd.DataFrame(mu, index=sample_names, columns=gene_names),
        normed_means=pd.Series(
            np.concatenate([normed_means, nan_genes]), index=gene_names
        ),
        gene_kind=pd.Series(kind, index=gene_names, dtype=object),
        outliers=pd.DataFrame(outliers, index=sample_names, columns=gene_names),
        trend=(float(trend[0]), float(trend[1])),
    )


# Hand-worked inputs


def size_factor_example() -> dict:
    """Return a 3 x 7 count matrix whose median-of-ratios size factors are exact.

    Genes ``g1``-``g5`` have counts ``2**e`` with exponents (samples x genes)::

             g1  g2  g3  g4  g5
        s1    0   1   3   2   4
        s2    1   1   0   2   1
        s3    2   4   3   2   1

    Column means of the exponents are 1, 2, 2, 2, 2, so the log2 ratios to the
    geometric means are s1: (-1, -1, 1, 0, 2), s2: (0, -1, -2, 0, -1),
    s3: (1, 2, 1, 0, -1). Their medians are 0, -1 and 1, giving size factors
    1, 0.5 and 2. ``g6`` has a zero and ``g7`` is all zero, so the ratio method
    ignores both. With control genes ``g1``-``g3`` the medians are -1, -1 and 1,
    giving 0.5, 0.5 and 2.

    Returns
    -------
    Dict with ``counts``, ``metadata``, ``ratio_size_factors``, ``control_genes`` and
    ``control_size_factors``.
    """
    exponents = np.array([[0, 1, 3, 2, 4], [1, 1, 0, 2, 1], [2, 4, 3, 2, 1]])
    counts = np.hstack(
        [2**exponents, np.array([[0], [5], [7]]), np.zeros((3, 1), dtype=int)]
    )
    samples = ["s1", "s2", "s3"]
    return {
        "counts": pd.DataFrame(
            counts, index=samples, columns=[f"g{i}" for i in range(1, 8)]
        ),
        "metadata": pd.DataFrame({"condition": ["A", "A", "B"]}, index=samples),
        "ratio_size_factors": np.array([1.0, 0.5, 2.0]),
        "control_genes": ["g1", "g2", "g3"],
        "control_size_factors": np.array([0.5, 0.5, 2.0]),
    }


def nb_example() -> dict:
    """Return a negative binomial log-likelihood worked by hand.

    With ``mu = 2`` and ``alpha = 1`` the NB is geometric with success probability
    ``p = 1 / (1 + alpha * mu) = 1/3``, so ``P(y) = (1/3) (2/3)**y``. For
    ``y = (0, 1, 3)`` the likelihood is ``(1/3) (2/9) (8/81) = 16/2187`` and the
    negative log-likelihood is ``log(2187/16) = 7 log 3 - 4 log 2``.
    """
    return {
        "counts": np.array([0, 1, 3]),
        "mu": np.array([2.0, 2.0, 2.0]),
        "alpha": 1.0,
        "nll": 7 * np.log(3) - 4 * np.log(2),
    }


# Reference likelihood and reference fits


def nb_logpmf(counts, mu, alpha):
    """Elementwise NB log-pmf with mean ``mu`` and variance ``mu + alpha * mu**2``.

    This is ``scipy.stats.nbinom.logpmf(counts, 1/alpha, 1/(1 + alpha*mu))``
    rewritten so that it stays accurate as ``alpha`` goes to zero: with
    ``r = 1/alpha``, ``log C(y + r - 1, y) = -log(y) - betaln(r, y)`` and the
    remaining terms use ``log1p(alpha*mu)``. scipy's own form subtracts
    ``gammaln(r + y) - gammaln(r)`` and multiplies ``log(p)`` by ``r``, which is off
    by up to 2e-7 per sample at ``alpha=1e-8``. The self-tests check this
    function against ``scipy.stats.nbinom`` (to 1e-12 for ``alpha >= 0.01``, to
    scipy's own 1e-10 at ``alpha=1e-4``), the Poisson limit and hand-worked values. It shares no code with PyDESeq2's
    ``nb_nll``. Arguments broadcast.
    """
    y, mu, alpha = np.broadcast_arrays(
        np.asarray(counts, dtype=float),
        np.asarray(mu, dtype=float),
        np.asarray(alpha, dtype=float),
    )
    positive = y > 0
    y_safe = np.where(positive, y, 1.0)
    log_comb = np.where(
        positive, -np.log(y_safe) - special.betaln(1.0 / alpha, y_safe), 0.0
    )
    log1p_am = np.log1p(alpha * mu)
    return log_comb - log1p_am / alpha + special.xlogy(y, alpha * mu) - y * log1p_am


def nb_loglik(counts, mu, alpha, axis=0):
    """NB log-likelihood summed over ``axis`` (samples, by default)."""
    return nb_logpmf(counts, mu, alpha).sum(axis=axis)


def dispersion_objective(
    counts: np.ndarray,
    mu: np.ndarray,
    design_matrix: np.ndarray,
    *,
    cr_reg: bool = True,
    prior_disp_var: float | None = None,
    prior_dispersion: float | None = None,
):
    """Return the dispersion objective of one gene as a function of log dispersion.

    The objective is DESeq2's Cox-Reid adjusted profile likelihood, negated, with an
    optional normal prior on the log dispersion:
    ``-loglik(alpha) + 0.5 log det(X^T W X) + (log alpha - log prior)^2 / (2 var)``,
    with ``W = mu / (1 + alpha mu)``. The returned function accepts an array of log
    dispersions and evaluates them all at once.
    """
    counts = np.asarray(counts, dtype=float)
    mu = np.asarray(mu, dtype=float)
    x = np.asarray(design_matrix, dtype=float)
    if prior_disp_var is not None and prior_dispersion is None:
        raise ValueError("prior_dispersion is required with prior_disp_var.")

    def objective(log_alpha):
        log_alpha = np.asarray(log_alpha, dtype=float)
        alpha = np.exp(log_alpha)
        flat = np.atleast_1d(alpha)
        value = -nb_loglik(counts[:, None], mu[:, None], flat[None, :], axis=0)
        if cr_reg:
            w = mu[None, :] / (1.0 + flat[:, None] * mu[None, :])
            xtwx = np.einsum("ip,ki,iq->kpq", x, w, x)
            value = value + 0.5 * np.linalg.slogdet(xtwx)[1]
        if prior_disp_var is not None:
            value = value + (np.log(flat) - np.log(prior_dispersion)) ** 2 / (
                2 * prior_disp_var
            )
        return value.reshape(log_alpha.shape) if log_alpha.ndim else float(value[0])

    return objective


def reference_dispersion_mle(
    counts: np.ndarray,
    mu: np.ndarray,
    design_matrix: np.ndarray,
    *,
    min_disp: float = 1e-8,
    max_disp: float = 10.0,
    cr_reg: bool = True,
    prior_disp_var: float | None = None,
    prior_dispersion: float | None = None,
    grid_size: int = 2001,
) -> float:
    """Minimize :func:`dispersion_objective` over ``[min_disp, max_disp]``.

    Evaluates the objective on ``grid_size`` log-spaced points (spacing 0.01 in log
    dispersion for the default bounds), then refines the best point with a bounded
    Brent search between its neighbours (``xatol`` 1e-10). The grid makes the result
    independent of the starting point and finds boundary minima. Keep in mind that
    PyDESeq2 clips at ``max(max_disp, n_samples)``, not ``max_disp``.
    """
    objective = dispersion_objective(
        counts,
        mu,
        design_matrix,
        cr_reg=cr_reg,
        prior_disp_var=prior_disp_var,
        prior_dispersion=prior_dispersion,
    )
    grid = np.linspace(np.log(min_disp), np.log(max_disp), grid_size)
    values = objective(grid)
    i = int(np.nanargmin(values))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid_size - 1)]
    res = optimize.minimize_scalar(
        objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10}
    )
    best = res.x if res.fun <= values[i] else grid[i]
    return float(np.exp(best))


def glm_objective(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    disp: float,
    *,
    ridge: float = 1e-6,
):
    """Return the penalized NB GLM objective of one gene as a function of beta.

    ``-loglik(counts | size_factors * exp(X beta), disp) + 0.5 * ridge * |beta|^2``,
    with beta in natural log scale. ``ridge=1e-6`` is the penalty PyDESeq2's IRLS
    adds (``ridge_factor``), so the minimizer is the fixed point of its iterations
    whenever no fitted mean falls below ``min_mu``.
    """
    counts = np.asarray(counts, dtype=float)
    sf = np.asarray(size_factors, dtype=float)
    x = np.asarray(design_matrix, dtype=float)

    def objective(beta):
        mu = sf * np.exp(x @ np.asarray(beta, dtype=float))
        return float(-nb_loglik(counts, mu, disp) + 0.5 * ridge * np.sum(beta**2))

    return objective


def reference_glm_mle(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    disp: float,
    *,
    ridge: float = 1e-6,
    max_iter: int = 200,
) -> np.ndarray:
    """Minimize :func:`glm_objective` by Newton's method with step halving.

    The objective is strictly convex in beta (its Hessian is
    ``X^T diag(mu (1 + alpha y) / (1 + alpha mu)^2) X + ridge I``), so Newton's
    method from a least-squares start reaches the unique minimizer. Stops after the
    first Newton step below 1e-10 in every coordinate; Newton converges
    quadratically, so the error left after that step is at the level of rounding.
    Raises if that does not happen in ``max_iter`` steps.
    """
    y = np.asarray(counts, dtype=float)
    sf = np.asarray(size_factors, dtype=float)
    x = np.asarray(design_matrix, dtype=float)
    objective = glm_objective(y, sf, x, disp, ridge=ridge)
    beta = np.linalg.lstsq(x, np.log(y / sf + 0.1), rcond=None)[0]
    eye = np.eye(x.shape[1])
    for _ in range(max_iter):
        mu = sf * np.exp(x @ beta)
        grad = -x.T @ ((y - mu) / (1.0 + disp * mu)) + ridge * beta
        weights = mu * (1.0 + disp * y) / (1.0 + disp * mu) ** 2
        hess = (x.T * weights) @ x + ridge * eye
        step = np.linalg.solve(hess, grad)
        if np.max(np.abs(step)) < 1e-10:
            return beta - step
        # Halve the step while it increases the objective by more than rounding.
        f0, t = objective(beta), 1.0
        while objective(beta - t * step) > f0 + 1e-12 * abs(f0) and t > 1e-12:
            t /= 2
        beta = beta - t * step
    raise RuntimeError("reference_glm_mle did not converge.")


# Comparison


@dataclass
class FieldDifference:
    """How one field (column) of ``actual`` differs from ``expected``."""

    field: str
    n_compared: int
    n_failed: int
    nan_mismatch: list
    max_abs_diff: float
    max_rel_diff: float
    worst: list  # (label, actual, expected, rel_diff), worst first


def _as_frame(values, name: str) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values
    if isinstance(values, pd.Series):
        return values.to_frame(name=values.name if values.name is not None else name)
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        return pd.DataFrame({name: arr})
    return pd.DataFrame(arr.reshape(arr.shape[0], -1))


def compare_fields(
    actual,
    expected,
    *,
    rtol: float,
    atol: float = 0.0,
    name: str = "value",
    n_worst: int = 5,
) -> list[FieldDifference]:
    """Compare two tables field by field with ``|a - e| <= atol + rtol * |e|``.

    Accepts DataFrames, Series or arrays (rows are genes or samples, columns are
    fields). Shapes must match. When both sides are labeled, the labels must match
    exactly; an array compared with a labeled table takes the table's labels. NaN
    positions must match exactly (both NaN counts as equal), and so must infinities.

    Returns
    -------
    One :class:`FieldDifference` per column, including columns that pass.
    """
    a = _as_frame(actual, name)
    e = _as_frame(expected, name)
    a_labeled = isinstance(actual, (pd.Series, pd.DataFrame))
    e_labeled = isinstance(expected, (pd.Series, pd.DataFrame))
    if a.shape != e.shape:
        raise AssertionError(f"{name}: shapes differ, {a.shape} vs {e.shape}.")
    if a_labeled != e_labeled:
        # An array compared with a labeled table takes the table's labels.
        labeled, unlabeled = (a, e) if a_labeled else (e, a)
        unlabeled.index, unlabeled.columns = labeled.index, labeled.columns
    elif a_labeled:
        if not a.index.equals(e.index):
            raise AssertionError(
                f"{name}: row labels differ.\n  actual:   {list(a.index)[:10]}"
                f"\n  expected: {list(e.index)[:10]}"
            )
        if isinstance(actual, pd.Series) and isinstance(expected, pd.Series):
            e.columns = a.columns
        if not a.columns.equals(e.columns):
            raise AssertionError(
                f"{name}: fields differ: actual {list(a.columns)}, "
                f"expected {list(e.columns)}."
            )

    out = []
    for col_a, col_e in zip(a.columns, e.columns, strict=True):
        av = np.asarray(a[col_a], dtype=float)
        ev = np.asarray(e[col_e], dtype=float)
        labels = np.asarray(a.index)
        nan_a, nan_e = np.isnan(av), np.isnan(ev)
        nan_mismatch = list(labels[nan_a != nan_e])
        both = ~nan_a & ~nan_e
        inf_mismatch = both & (np.isinf(av) | np.isinf(ev)) & (av != ev)
        finite = both & np.isfinite(av) & np.isfinite(ev)
        abs_diff = np.zeros_like(av)
        abs_diff[finite] = np.abs(av[finite] - ev[finite])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.where(finite, abs_diff / np.abs(ev), 0.0)
        rel_diff[finite & (abs_diff == 0)] = 0.0
        rel_diff[inf_mismatch] = np.inf
        failed = (finite & (abs_diff > atol + rtol * np.abs(ev))) | inf_mismatch
        order = np.argsort(-np.where(failed, rel_diff, -1.0), kind="stable")
        worst = [
            (labels[i], av[i], ev[i], rel_diff[i]) for i in order[:n_worst] if failed[i]
        ]
        out.append(
            FieldDifference(
                field=str(col_a),
                n_compared=int(both.sum()),
                n_failed=int(failed.sum()),
                nan_mismatch=nan_mismatch,
                max_abs_diff=float(abs_diff[finite].max()) if finite.any() else 0.0,
                max_rel_diff=float(np.nanmax(rel_diff[both])) if both.any() else 0.0,
                worst=worst,
            )
        )
    return out


def assert_fields_close(
    actual,
    expected,
    *,
    rtol: float,
    atol: float = 0.0,
    name: str = "value",
    n_worst: int = 5,
) -> None:
    """Assert :func:`compare_fields` finds no difference; the message names them.

    The message lists, per failing field, how many entries fail, the NaN mismatches
    and the worst entries by relative difference with their labels.
    """
    diffs = compare_fields(
        actual, expected, rtol=rtol, atol=atol, name=name, n_worst=n_worst
    )
    lines = []
    for d in diffs:
        if d.n_failed == 0 and not d.nan_mismatch:
            continue
        lines.append(
            f"{name}[{d.field}]: {d.n_failed}/{d.n_compared} outside "
            f"rtol={rtol:g}, atol={atol:g}; max rel diff {d.max_rel_diff:.3g}, "
            f"max abs diff {d.max_abs_diff:.3g}"
        )
        if d.nan_mismatch:
            lines.append(f"  NaN in one but not the other at: {d.nan_mismatch[:10]}")
        for label, av, ev, rel in d.worst:
            lines.append(f"  {label}: actual {av!r}, expected {ev!r}, rel {rel:.3g}")
    if lines:
        raise AssertionError("\n".join(lines))
