"""Area B: dispersions (see tests/TARGETS.md)."""

import numpy as np
import pytest

from pydeseq2.dispersions import fit_alpha_mle
from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS

# fit_alpha_mle stops L-BFGS-B at a projected gradient of 1e-5 in log dispersion,
# so its error is about 1e-5 over the curvature of the objective. Against the
# reference minimizer (Cox-Reid on) that left relative errors up to 9.5e-5 on 450
# simulated genes with 6-40 samples; 1e-3 leaves a 10x margin.
DISPERSION_RTOL = 1e-3


@pytest.mark.parametrize(
    "n_samples, seed, gene",
    [(6, 1, "gene1"), (6, 1, "gene2"), (12, 2, "gene3"), (40, 3, "gene4")],
)
def test_fit_alpha_mle_matches_reference(n_samples, seed, gene):
    """One gene's Cox-Reid adjusted dispersion MLE, given the true means.

    Without Cox-Reid the objective of near-Poisson genes is flat enough that
    L-BFGS-B stops far from the minimizer (relative errors above 100 with objective
    gaps below 1e-5), so such tests need an objective-gap criterion instead.
    """
    sim = helpers.simulate_counts(
        n_genes=4, n_samples=n_samples, seed=seed, disp_scatter=0.5
    )
    x = sim.design_matrix.to_numpy()
    y, mu = sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy()

    alpha, converged = fit_alpha_mle(
        counts=y,
        design_matrix=x,
        mu=mu,
        alpha_hat=0.1,
        min_disp=1e-8,
        max_disp=10.0,
    )
    expected = helpers.reference_dispersion_mle(y, mu, x, min_disp=1e-8, max_disp=10.0)

    assert converged
    assert alpha == pytest.approx(expected, rel=DISPERSION_RTOL)


def test_fit_alpha_mle_with_prior_matches_reference():
    """MAP dispersion: the prior is centred on ``alpha_hat`` (the trend value)."""
    sim = helpers.simulate_counts(n_genes=3, n_samples=8, seed=4, disp_scatter=0.5)
    x = sim.design_matrix.to_numpy()
    for gene in sim.counts.columns:
        y, mu = sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy()
        prior = 0.5 * sim.dispersions[gene]
        alpha, converged = fit_alpha_mle(
            counts=y,
            design_matrix=x,
            mu=mu,
            alpha_hat=prior,
            min_disp=1e-8,
            max_disp=10.0,
            prior_disp_var=0.25,
            cr_reg=True,
            prior_reg=True,
        )
        expected = helpers.reference_dispersion_mle(
            y, mu, x, prior_disp_var=0.25, prior_dispersion=prior
        )
        assert converged
        assert alpha == pytest.approx(expected, rel=DISPERSION_RTOL), gene


def test_genewise_dispersions_match_reference():
    """``var["genewise_dispersions"]`` against the reference, gene by gene.

    With a two-group design PyDESeq2 takes the means from a linear model on the
    normalized counts, which reduces to the group means times the size factor,
    floored at ``min_mu=0.5``; the reference recomputes them that way. Dispersions
    are clipped to ``[min_disp, max(max_disp, n_samples)]``.

    Genes whose starting value, DESeq2's minimum of the rough and moments estimates,
    is at ``min_disp`` are left out: the objective is flat in log dispersion there,
    and PyDESeq2, like DESeq2 1.50.2, stays at ``min_disp`` even when the minimizer
    is elsewhere (for example 1e-8 instead of 0.034 for counts 18, 2, 10, 4, 14, 6,
    19, 7, 5, 5 in two alternating groups).
    """
    sim = helpers.simulate_counts(n_genes=60, n_samples=10, seed=5, disp_scatter=0.3)
    dds = helpers.make_dds(sim.counts, sim.metadata)
    dds.fit_size_factors()
    dds.fit_genewise_dispersions()

    sf = dds.obs["size_factors"].to_numpy()
    normed = sim.counts.to_numpy() / sf[:, None]
    groups = sim.metadata["condition"].to_numpy()
    group_means = np.empty_like(normed)
    for level in np.unique(groups):
        group_means[groups == level] = normed[groups == level].mean(axis=0)
    mu_hat = np.maximum(sf[:, None] * group_means, 0.5)

    # DESeq2's roughDispEstimate and momentsDispEstimate.
    n, p = sim.design_matrix.shape
    fitted = np.maximum(group_means, 1.0)
    rough = (((normed - fitted) ** 2 - fitted) / fitted**2).sum(axis=0) / (n - p)
    means = normed.mean(axis=0)
    moments = (normed.var(axis=0, ddof=1) - (1 / sf).mean() * means) / means**2
    keep = np.minimum(rough, moments) > 1e-8
    assert keep.sum() >= 50

    x = sim.design_matrix.to_numpy()
    max_disp = max(10.0, n)
    expected = np.array(
        [
            helpers.reference_dispersion_mle(
                sim.counts.iloc[:, j].to_numpy(), mu_hat[:, j], x, max_disp=max_disp
            )
            for j in np.flatnonzero(keep)
        ]
    )
    helpers.assert_fields_close(
        dds.var["genewise_dispersions"][keep],
        expected,
        rtol=DISPERSION_RTOL,
        name="genewise_dispersions",
    )
