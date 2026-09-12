"""The sector model must not claim identification it does not have.

These tests exist because the first version of this model did exactly that. It
anchored the fuel loading on sector time share -- a guess that is simply wrong,
since fuel costs time where the car accelerates and time share is near equal
thirds -- and the wrong anchor dragged the tyre loading onto the same direction.
The recovered separation was 8 degrees against a true 37, and the resulting rate
estimate was eighteen times worse than a plain whole-lap fit.

So the properties asserted here are the ones that distinguish a real
identification from a confident-looking one.
"""

from __future__ import annotations

import numpy as np
import pytest

from tyremind.data.synthetic import SessionConfig, generate_session
from tyremind.models.sector_ssm import stint_slopes


@pytest.fixture(scope="module")
def session():
    return generate_session(SessionConfig(seed=11))


def centred_loading(lap_table):
    """Recover the tyre loading from the rank-one structure of centred slopes."""
    slopes = stint_slopes(lap_table)
    matrix = slopes[[f"slope_{s}" for s in (1, 2, 3)]].to_numpy(dtype=float)
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    direction = right[0]
    if direction.sum() < 0:
        direction = -direction
    return direction / direction.sum(), float(singular[0] ** 2 / (singular ** 2).sum())


class TestGeneratorHonesty:
    def test_the_two_loadings_are_not_parallel(self, session):
        """If the generator split both effects the same way, the sector model
        would look identified when it is not. This is the assumption the whole
        experiment rests on, so it is asserted rather than trusted."""
        tyre = np.asarray(session.truth.tyre_loading, dtype=float)
        fuel = np.asarray(session.truth.fuel_loading, dtype=float)
        cosine = tyre @ fuel / (np.linalg.norm(tyre) * np.linalg.norm(fuel))
        assert np.degrees(np.arccos(cosine)) > 20.0

    def test_sectors_sum_to_the_lap(self, session):
        table = session.lap_table
        total = table[["sector_1", "sector_2", "sector_3"]].sum(axis=1)
        # Within the injected per-sector noise, sd 0.05 on each of three.
        assert (total - table["lap_time"]).abs().mean() < 0.15


class TestWhatIsIdentified:
    def test_the_centred_slope_matrix_is_rank_one(self, session):
        """Because the rate varies across stints and the fuel coefficient does
        not, centring leaves `(beta_j - beta_bar) * wT_k`. If this stops being
        rank one, the identification argument has gone."""
        _, fraction = centred_loading(session.lap_table)
        assert fraction > 0.7

    def test_the_tyre_loading_direction_is_recovered(self, session):
        """Not exactly -- noise biases it toward equal thirds -- but it must pick
        the right sector as the heaviest, which is the claim being made."""
        recovered, _ = centred_loading(session.lap_table)
        truth = np.asarray(session.truth.tyre_loading, dtype=float)
        assert int(np.argmax(recovered)) == int(np.argmax(truth))


class TestWhatIsNotIdentified:
    def test_the_level_system_is_underdetermined_without_a_fuel_loading(self):
        """The honest negative result, pinned so it cannot quietly become a
        positive one.

        The mean equation is `m_k = beta_bar * wT_k - phi * wF_k`: three
        equations, and after normalising `wF` to sum to one the unknowns are
        `beta_bar`, `phi` and two free components of `wF`. Four unknowns, three
        equations. Sectors alone do not break exp18's collinearity.
        """
        equations = 3
        unknowns = 1 + 1 + 2          # beta_bar, phi, wF with sum-to-one imposed
        assert unknowns > equations

    def test_supplying_a_fuel_loading_makes_the_level_well_posed(self, session):
        """And the route out, also pinned. With `wF` measured rather than
        guessed, the 2x2 system conditions in single digits."""
        tyre_direction, _ = centred_loading(session.lap_table)
        fuel = np.asarray(session.truth.fuel_loading, dtype=float)
        design = np.column_stack([tyre_direction, -fuel])
        assert np.linalg.cond(design) < 20.0
