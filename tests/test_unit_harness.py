"""Self-tests of the shared helpers in ``tests/helpers.py``."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS


def test_get_test_cpus(monkeypatch):
    monkeypatch.delenv(helpers.CPUS_ENV_VAR, raising=False)
    assert helpers.get_test_cpus() == 1
    monkeypatch.setenv(helpers.CPUS_ENV_VAR, "4")
    assert helpers.get_test_cpus() == 4
    monkeypatch.setenv(helpers.CPUS_ENV_VAR, "0")
    with pytest.raises(ValueError):
        helpers.get_test_cpus()


def test_n_cpus_fixtures(n_cpus, inference):
    assert n_cpus == helpers.get_test_cpus()
    assert inference.n_cpus == n_cpus
    assert helpers.make_dds(**_tiny_counts()).inference.n_cpus == n_cpus


def _tiny_counts():
    sim = helpers.simulate_counts(n_genes=5, n_samples=4, seed=0)
    return {"counts": sim.counts, "metadata": sim.metadata}


def test_simulate_counts_is_reproducible():
    """Same seed, same counts; the counts are integers, so equality is exact."""
    kwargs = {"n_all_zero": 2, "n_single_sample": 2, "n_large": 2, "n_outliers": 3}
    first = helpers.simulate_counts(seed=7, **kwargs)
    second = helpers.simulate_counts(seed=7, **kwargs)
    other = helpers.simulate_counts(seed=8, **kwargs)
    pd.testing.assert_frame_equal(first.counts, second.counts)
    assert not first.counts.equals(other.counts)


def test_simulate_counts_special_genes_leave_regular_genes_unchanged():
    plain = helpers.simulate_counts(n_genes=30, seed=4)
    extra = helpers.simulate_counts(
        n_genes=30, seed=4, n_all_zero=2, n_single_sample=2, n_large=3, n_outliers=5
    )
    regular = plain.counts.columns
    outliers = extra.outliers[regular].to_numpy()
    np.testing.assert_array_equal(
        extra.counts[regular].to_numpy()[~outliers], plain.counts.to_numpy()[~outliers]
    )
    pd.testing.assert_frame_equal(extra.beta.loc[regular], plain.beta)


def test_simulate_counts_pins_first_counts():
    """Pins the current output: the first counts drawn for seed 0.

    The simulator draws from legacy ``RandomState`` streams, which NumPy keeps
    fixed across versions; this test fails if that or the simulator changes, which
    would silently change every test built on simulated data.
    """
    sim = helpers.simulate_counts(n_genes=3, n_samples=4, seed=0)
    expected = pd.DataFrame(
        [[14, 815, 9], [11, 499, 16], [5, 906, 5], [4, 497, 11]],
        index=["sample1", "sample2", "sample3", "sample4"],
        columns=["gene1", "gene2", "gene3"],
    )
    pd.testing.assert_frame_equal(sim.counts, expected)


def test_simulate_counts_truth_is_consistent():
    sim = helpers.simulate_counts(
        n_genes=50,
        n_samples=12,
        seed=3,
        factors={"condition": ["A", "B"], "group": ["X", "Y", "Z"]},
        continuous=("age",),
        n_all_zero=2,
        n_single_sample=3,
        n_large=2,
        n_outliers=4,
    )
    regular = sim.gene_kind.isin(["null", "de", "large"]).to_numpy()
    special = ~regular
    x = sim.design_matrix.to_numpy()

    # Shapes and labels line up.
    assert sim.counts.shape == (12, 50 + 2 + 2 + 3)
    assert list(sim.beta.index) == list(sim.counts.columns)
    assert list(sim.beta.columns) == list(sim.design_matrix.columns)
    assert sim.counts.dtypes.eq(np.int64).all()

    # mu is size factors times exp(design @ beta); 1e-12 covers exp/matmul rounding.
    expected_mu = sim.size_factors.to_numpy()[:, None] * np.exp(
        x @ sim.beta.to_numpy()[regular].T
    )
    np.testing.assert_allclose(sim.mu.to_numpy()[:, regular], expected_mu, rtol=1e-12)
    np.testing.assert_allclose(
        sim.normed_means[regular],
        (expected_mu / sim.size_factors.to_numpy()[:, None]).mean(0),
        rtol=1e-12,
    )
    assert np.exp(np.log(sim.size_factors).mean()) == pytest.approx(1.0, rel=1e-12)

    # Without scatter, dispersions sit on the trend.
    a0, a1 = sim.trend
    np.testing.assert_allclose(
        sim.dispersions[regular], a0 + a1 / sim.normed_means[regular], rtol=1e-12
    )

    # Non-DE genes have zero non-intercept coefficients.
    null = (sim.gene_kind == "null").to_numpy()
    assert (sim.beta.to_numpy()[null, 1:] == 0).all()
    assert (
        (sim.beta.to_numpy()[(sim.gene_kind == "de").to_numpy(), 1:] != 0)
        .any(axis=1)
        .all()
    )

    # Special genes.
    counts = sim.counts.to_numpy()
    assert (counts[:, (sim.gene_kind == "all_zero").to_numpy()] == 0).all()
    single = counts[:, (sim.gene_kind == "single_sample").to_numpy()]
    assert ((single > 0).sum(axis=0) == 1).all()
    assert (
        sim.beta.to_numpy()[special].size
        and np.isnan(sim.beta.to_numpy()[special]).all()
    )
    assert np.isnan(sim.dispersions.to_numpy()[special]).all()

    # Outliers: one per gene, at least outlier_factor times the mean.
    assert sim.outliers.to_numpy().sum() == 4
    assert (sim.outliers.sum(axis=0) <= 1).all()
    mask = sim.outliers.to_numpy()
    assert (counts[mask] >= 100 * np.maximum(sim.mu.to_numpy()[mask], 1)).all()


def test_simulate_counts_design_matches_deseq_dataset():
    """The simulator's design matrix is the one PyDESeq2 builds from the formula."""
    sim = helpers.simulate_counts(
        n_genes=10,
        n_samples=12,
        seed=1,
        factors={"condition": ["A", "B"], "group": ["X", "Y", "Z"]},
        continuous=("age",),
    )
    dds = helpers.make_dds(sim.counts, sim.metadata, design=sim.design)
    dm = dds.obsm["design_matrix"]
    assert list(dm.columns) == list(sim.design_matrix.columns)
    # 0/1 indicators and copied covariates: exact by construction.
    np.testing.assert_array_equal(dm.to_numpy(dtype=float), sim.design_matrix.to_numpy())


def test_simulate_counts_moments():
    """Sample mean and variance match mu and mu + alpha mu^2 within 5 standard errors.

    The standard errors are estimated from the same draws (for the variance, the
    standard deviation of the squared deviations over sqrt(n)), so the bound is a
    5-sigma statistical tolerance, not a fitted one.
    """
    n = 4000
    sim = helpers.simulate_counts(
        n_genes=3, n_samples=n, seed=11, frac_de=0.0, size_factor_sd=0.0
    )
    for gene in sim.counts.columns:
        y = sim.counts[gene].to_numpy(dtype=float)
        mu = sim.mu[gene].iloc[0]
        alpha = sim.dispersions[gene]
        var = mu + alpha * mu**2
        assert abs(y.mean() - mu) < 5 * np.sqrt(var / n)
        sq = (y - y.mean()) ** 2
        assert abs(sq.mean() - var) < 5 * sq.std() / np.sqrt(n)


@pytest.mark.parametrize(
    "counts, mu, alpha, nll",
    [
        # Geometric, p = 1/3: P(y) = (1/3)(2/3)^y; y = 0, 1, 3 gives 16/2187.
        ([0, 1, 3], 2.0, 1.0, 7 * np.log(3) - 4 * np.log(2)),
        # n = 2, p = 1/2: P(y) = (y + 1) / 4 * (1/2)^y; y = 0, 2 gives 3/64.
        ([0, 2], 2.0, 0.5, np.log(64 / 3)),
    ],
)
def test_nb_loglik_hand_worked(counts, mu, alpha, nll):
    """1e-13 relative: the closed forms differ from scipy only by rounding."""
    mu = np.full(len(counts), mu)
    assert -helpers.nb_loglik(np.array(counts), mu, alpha) == pytest.approx(
        nll, rel=1e-13
    )


@pytest.mark.parametrize(
    "alpha, rtol", [(1e-4, 1e-10), (0.01, 1e-12), (0.5, 1e-12), (3.0, 1e-12)]
)
def test_nb_logpmf_matches_scipy(alpha, rtol):
    """Agree with scipy.stats.nbinom to scipy's own precision.

    scipy subtracts gammaln(1/alpha + y) - gammaln(1/alpha), which loses about one
    ulp of gammaln(1/alpha): ~1e-11 absolute at alpha=1e-4, hence 1e-10 there.
    """
    y = np.array([0, 1, 2, 5, 17, 140, 3000])
    mu = np.array([0.3, 1.0, 4.0, 5.5, 20.0, 100.0, 2500.0])
    expected = stats.nbinom.logpmf(y, 1 / alpha, 1 / (1 + alpha * mu))
    np.testing.assert_allclose(helpers.nb_logpmf(y, mu, alpha), expected, rtol=rtol)


def test_nb_logpmf_poisson_limit():
    """At alpha=1e-14 the NB is Poisson to ~alpha * y^2 relative (< 1e-9 here)."""
    y = np.array([0, 1, 2, 5, 17, 140])
    mu = np.array([0.3, 1.0, 4.0, 5.5, 20.0, 100.0])
    np.testing.assert_allclose(
        helpers.nb_logpmf(y, mu, 1e-14), stats.poisson.logpmf(y, mu), rtol=1e-9
    )


def test_nb_example():
    example = helpers.nb_example()
    assert -helpers.nb_loglik(
        example["counts"], example["mu"], example["alpha"]
    ) == pytest.approx(example["nll"], rel=1e-13)


def test_reference_dispersion_mle_is_a_minimum():
    sim = helpers.simulate_counts(n_genes=20, n_samples=8, seed=5, disp_scatter=0.5)
    x = sim.design_matrix.to_numpy()
    for gene in sim.counts.columns:
        y, mu = sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy()
        objective = helpers.dispersion_objective(y, mu, x)
        log_alpha = np.log(helpers.reference_dispersion_mle(y, mu, x))
        # Brent's xatol is 1e-10; a step of 1e-3 changes the objective by far more
        # than rounding, so neither side may be lower.
        best = objective(log_alpha)
        assert best <= objective(log_alpha - 1e-3)
        assert best <= objective(log_alpha + 1e-3)


def test_reference_dispersion_mle_finds_boundary():
    """Under-dispersed counts without Cox-Reid push the MLE to ``min_disp``."""
    y = np.array([10, 10, 10, 10, 10, 10])
    mu = np.full(6, 10.0)
    x = np.ones((6, 1))
    alpha = helpers.reference_dispersion_mle(y, mu, x, cr_reg=False, min_disp=1e-8)
    assert alpha == pytest.approx(1e-8, rel=1e-9)


def test_reference_dispersion_objective_prior():
    """The prior term is (log a - log a0)^2 / (2 var), checked at one point."""
    y, mu, x = np.array([3, 7, 1, 9]), np.full(4, 5.0), np.ones((4, 1))
    plain = helpers.dispersion_objective(y, mu, x)
    with_prior = helpers.dispersion_objective(
        y, mu, x, prior_disp_var=0.5, prior_dispersion=0.2
    )
    log_a = np.log(0.3)
    expected = plain(log_a) + (log_a - np.log(0.2)) ** 2 / (2 * 0.5)
    assert with_prior(log_a) == pytest.approx(expected, rel=1e-14)


def test_reference_glm_mle_is_stationary():
    sim = helpers.simulate_counts(
        n_genes=20,
        n_samples=10,
        seed=6,
        factors={"condition": ["A", "B"]},
        continuous=("age",),
    )
    x, sf = sim.design_matrix.to_numpy(), sim.size_factors.to_numpy()
    for gene in sim.counts.columns:
        y, alpha = sim.counts[gene].to_numpy(), sim.dispersions[gene]
        beta = helpers.reference_glm_mle(y, sf, x, alpha)
        mu = sf * np.exp(x @ beta)
        grad = -x.T @ ((y - mu) / (1 + alpha * mu)) + 1e-6 * beta
        # Gradient of a sum of ~10 terms of size ~y: 1e-8 relative to sum(y).
        assert np.max(np.abs(grad)) < 1e-8 * y.sum()
        objective = helpers.glm_objective(y, sf, x, alpha)
        for k in range(len(beta)):
            step = np.zeros_like(beta)
            step[k] = 1e-5
            assert objective(beta) <= objective(beta + step)
            assert objective(beta) <= objective(beta - step)


def test_compare_fields_names_the_differences():
    genes = ["g1", "g2", "g3"]
    expected = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [1.0, 1.0, 1.0]}, index=genes)
    actual = expected.copy()
    helpers.assert_fields_close(actual, expected, rtol=0)

    actual.loc["g2", "b"] = 1.1
    actual.loc["g3", "a"] = 5.0
    with pytest.raises(AssertionError) as excinfo:
        helpers.assert_fields_close(actual, expected, rtol=1e-3, name="res")
    message = str(excinfo.value)
    assert "res[b]: 1/3" in message and "g2" in message
    assert "res[a]" in message and "NaN in one but not the other at: ['g3']" in message
    helpers.assert_fields_close(actual["b"], expected["b"], rtol=0.2)

    diffs = helpers.compare_fields(actual, expected, rtol=1e-3)
    assert [d.field for d in diffs] == ["a", "b"]
    assert diffs[1].max_rel_diff == pytest.approx(0.1)

    with pytest.raises(AssertionError, match="row labels differ"):
        helpers.assert_fields_close(
            actual, expected.rename(index={"g1": "x"}), rtol=1e-3
        )
    with pytest.raises(AssertionError, match="shapes differ"):
        helpers.assert_fields_close(np.ones(3), np.ones(4), rtol=1e-3)
