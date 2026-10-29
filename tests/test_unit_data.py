"""Area A: data handling and normalization (see tests/TARGETS.md)."""

import numpy as np

from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS


def test_ratio_size_factors_hand_worked():
    """Median-of-ratios on the matrix worked in ``helpers.size_factor_example``.

    The expected values are exact powers of two; 1e-12 relative covers the rounding
    of log, mean, median and exp.
    """
    example = helpers.size_factor_example()
    dds = helpers.make_dds(example["counts"], example["metadata"])
    dds.fit_size_factors(fit_type="ratio")

    np.testing.assert_allclose(
        dds.obs["size_factors"], example["ratio_size_factors"], rtol=1e-12
    )
    np.testing.assert_allclose(
        dds.layers["normed_counts"],
        example["counts"].to_numpy() / example["ratio_size_factors"][:, None],
        rtol=1e-12,
    )


def test_ratio_size_factors_control_genes_hand_worked():
    """Control genes g1-g3 give medians -1, -1, 1 (``helpers.size_factor_example``)."""
    example = helpers.size_factor_example()
    dds = helpers.make_dds(
        example["counts"],
        example["metadata"],
        control_genes=example["control_genes"],
    )
    dds.fit_size_factors(fit_type="ratio")

    np.testing.assert_allclose(
        dds.obs["size_factors"], example["control_size_factors"], rtol=1e-12
    )
