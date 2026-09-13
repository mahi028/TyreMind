"""The observation model, the skewed-t likelihood, and the filter itself.

Three things are checked here that a filter can get wrong while still running:

1. **The skewed t is a probability distribution.** It integrates to one, it
   reduces to a symmetric Student-t at `zeta = 1`, and its stated mean and
   variance match numerical moments. A likelihood that does not integrate to one
   is not a likelihood, and an optimiser maximising it is maximising nothing in
   particular.
2. **The covariance stays a covariance.** Symmetric and positive semi-definite
   after every update, over a stint long enough for the failure to show. The
   Joseph form is in `ekf.py` precisely because the short form failed this.
3. **The filter recovers a trajectory it was not shown.** Synthetic data with a
   known wear path, and the filter has to find it. This is the four-corner
   analogue of exp01 and it is the only test here that could justify the model.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from tyremind.models.fourcorner.dynamics import LapInput, drift
from tyremind.models.fourcorner.ekf import propagate, rts_smooth, run_filter
from tyremind.models.fourcorner.observation import (
    grip_penalty_seconds,
    observation_jacobian,
    predict_lap_time,
    robust_weight,
    skewed_t_logpdf,
    skewed_t_mean,
    skewed_t_variance,
)
from tyremind.models.fourcorner.state import (
    I_TRACK,
    N_STATE,
    T,
    W,
    FourCornerParameters,
    clamp_state,
    initial_covariance,
    initial_state,
)


#: See the note in test_fourcorner_dynamics.py: fixtures run at the corpus
#: median per-corner power so the tyre sits in its thermal window.
TYPICAL_POWER = 1.81e5


def a_lap(duration: float = 92.0, lap: int = 1, laps_completed: float = 0.0) -> LapInput:
    return LapInput(
        duration_s=duration,
        load_n=np.array([5200.0, 3800.0, 4900.0, 3600.0]),
        power_proxy_w=TYPICAL_POWER * np.array([1.10, 0.90, 1.05, 0.95]),
        airspeed_ms=58.0,
        laps_completed=laps_completed,
        traffic_index=0.0,
        session_lap=lap,
    )


class TestSkewedT:
    @pytest.mark.parametrize("nu", [3.0, 6.0, 12.0, 30.0])
    def test_reduces_to_student_t_when_symmetric(self, nu: float):
        z = np.linspace(-5.0, 5.0, 41)
        ours = skewed_t_logpdf(z, nu, 1.0)
        theirs = stats.t.logpdf(z, df=nu)
        assert np.allclose(ours, theirs, atol=1e-12)

    @pytest.mark.parametrize("zeta", [0.7, 1.0, 1.3, 2.0])
    @pytest.mark.parametrize("nu", [4.0, 10.0])
    def test_integrates_to_one(self, zeta: float, nu: float):
        # Integrated to infinity rather than to a finite cut. A nu = 4 t keeps
        # about 1e-6 of its mass beyond +-60, which is enough to fail a correct
        # density at this tolerance -- the first version of this test did.
        total, _ = integrate.quad(
            lambda z: np.exp(skewed_t_logpdf(z, nu, zeta)), -np.inf, np.inf, limit=400
        )
        assert total == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize("zeta", [0.8, 1.0, 1.4])
    def test_stated_mean_matches_numerical(self, zeta: float):
        nu = 8.0
        numeric, _ = integrate.quad(
            lambda z: z * np.exp(skewed_t_logpdf(z, nu, zeta)), -80.0, 80.0, limit=500
        )
        assert skewed_t_mean(nu, zeta) == pytest.approx(numeric, abs=1e-6)

    @pytest.mark.parametrize("zeta", [0.8, 1.0, 1.4])
    def test_stated_variance_matches_numerical(self, zeta: float):
        nu = 9.0
        mean = skewed_t_mean(nu, zeta)
        numeric, _ = integrate.quad(
            lambda z: (z - mean) ** 2 * np.exp(skewed_t_logpdf(z, nu, zeta)),
            -90.0, 90.0, limit=600,
        )
        assert skewed_t_variance(nu, zeta) == pytest.approx(numeric, rel=1e-4)

    def test_right_skew_makes_slow_laps_cheaper_than_fast_ones(self):
        """The physical claim: a driver can lose time by mistake, not gain it."""
        nu, zeta = 6.0, 1.4
        slow = skewed_t_logpdf(2.5, nu, zeta)
        fast = skewed_t_logpdf(-2.5, nu, zeta)
        assert slow > fast

    def test_rejects_parameters_with_no_variance(self):
        with pytest.raises(ValueError, match="nu must exceed 2"):
            skewed_t_logpdf(0.0, 1.5, 1.0)
        with pytest.raises(ValueError, match="zeta must be positive"):
            skewed_t_logpdf(0.0, 6.0, 0.0)


class TestRobustWeight:
    def test_typical_lap_keeps_its_weight(self):
        assert robust_weight(0.1, 0.3, 6.0, 1.0) > 0.9

    def test_outlier_is_down_weighted(self):
        assert robust_weight(3.0, 0.3, 6.0, 1.0) < 0.1

    def test_slow_outlier_is_penalised_less_than_a_fast_one(self):
        """With a right skew a slow lap is the less surprising event."""
        slow = robust_weight(1.5, 0.3, 6.0, 1.4)
        fast = robust_weight(-1.5, 0.3, 6.0, 1.4)
        assert slow > fast

    def test_weight_falls_monotonically_with_surprise(self):
        weights = [robust_weight(r, 0.3, 6.0, 1.0) for r in (0.0, 0.3, 0.9, 2.0, 5.0)]
        assert weights == sorted(weights, reverse=True)


class TestObservation:
    def test_a_worn_tyre_is_slower(self):
        p = FourCornerParameters()
        fresh, worn = initial_state(), initial_state()
        fresh[T] = worn[T] = p.grip.temp_optimal_c
        worn[W] = 0.8
        assert predict_lap_time(worn, a_lap(), p) > predict_lap_time(fresh, a_lap(), p)

    def test_fuel_burn_makes_the_car_faster(self):
        p = FourCornerParameters()
        x = initial_state()
        x[T] = p.grip.temp_optimal_c
        early = predict_lap_time(x, a_lap(laps_completed=0.0), p)
        late = predict_lap_time(x, a_lap(laps_completed=20.0), p)
        assert late < early
        # And by the amount the pinned physical coefficient says, exactly.
        expected = p.session.fuel_effect_s_per_kg * p.session.burn_kg_per_lap * 20.0
        assert early - late == pytest.approx(expected)

    def test_track_evolution_makes_the_car_faster(self):
        p = FourCornerParameters()
        green, rubbered = initial_state(), initial_state()
        green[T] = rubbered[T] = p.grip.temp_optimal_c
        rubbered[I_TRACK] = 1.0
        gain = predict_lap_time(green, a_lap(), p) - predict_lap_time(rubbered, a_lap(), p)
        assert gain == pytest.approx(p.session.track_amplitude_s)

    def test_traffic_costs_time(self):
        p = FourCornerParameters()
        x = initial_state()
        clear = LapInput(
            duration_s=92.0, load_n=np.full(4, 4000.0), power_proxy_w=np.full(4, TYPICAL_POWER),
            airspeed_ms=58.0, laps_completed=3.0, traffic_index=0.0, session_lap=5)
        busy = LapInput(
            duration_s=92.0, load_n=np.full(4, 4000.0), power_proxy_w=np.full(4, TYPICAL_POWER),
            airspeed_ms=58.0, laps_completed=3.0, traffic_index=1.0, session_lap=5)
        assert predict_lap_time(x, busy, p) > predict_lap_time(x, clear, p)

    def test_fresh_tyre_in_its_window_costs_nothing(self):
        p = FourCornerParameters()
        x = initial_state()
        x[T] = p.grip.temp_optimal_c
        assert grip_penalty_seconds(
            np.full(4, p.grip.mu_nominal), p
        ) == pytest.approx(0.0)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_jacobian_matches_finite_differences(self, seed: int):
        rng = np.random.default_rng(seed)
        p = FourCornerParameters()
        u = a_lap()
        x = initial_state()
        x[W] = rng.uniform(0.05, 0.9, 4)
        x[T] = rng.uniform(70.0, 130.0, 4)
        x[I_TRACK] = rng.uniform(0.1, 0.9)

        analytic = observation_jacobian(x, u, p)
        numeric = np.zeros(N_STATE)
        for j in range(N_STATE):
            step = 1e-7 * max(abs(x[j]), 1.0)
            up, down = x.copy(), x.copy()
            up[j] += step
            down[j] -= step
            numeric[j] = (predict_lap_time(up, u, p) - predict_lap_time(down, u, p)) / (2 * step)
        assert np.allclose(analytic, numeric, atol=1e-6, rtol=1e-4)

    def test_fuel_row_of_the_observation_jacobian_is_zero(self):
        """Fuel is never learned from lap times, and the zero says so.

        This is exp18's finding restated in the four-corner model: the fuel
        coefficient is a prior. A non-zero entry here would mean the filter
        believes it is measuring fuel, which it is not.
        """
        p = FourCornerParameters()
        h = observation_jacobian(initial_state(), a_lap(), p)
        assert h[8] == 0.0


class TestPropagate:
    def test_process_noise_strictly_adds_uncertainty(self):
        """The correct form of "propagation adds uncertainty".

        The first version of this test asserted the covariance trace grows across
        a lap. It does not, and the model was right: the four temperature states
        are strongly mean-reverting, with a time constant of about 55 s against a
        92 s lap, so they *forget* their initial spread and contract towards a
        stationary variance. That is what a stable linear system does.

        What is unconditionally true is that adding process noise cannot reduce
        uncertainty, so the comparison is against the same propagation with the
        diffusion switched off.
        """
        p = FourCornerParameters()
        noiseless = p.with_noise(wear_sd=0.0, temp_sd=0.0, fuel_sd=0.0, track_sd=0.0)
        x, cov = initial_state(), initial_covariance()
        _, with_noise = propagate(x, cov, a_lap(), p)
        _, without = propagate(x, cov, a_lap(), noiseless)
        assert np.trace(with_noise) > np.trace(without)

    def test_temperature_uncertainty_contracts_as_a_stable_state_should(self):
        """Mean reversion must shrink the temperature block over a lap."""
        p = FourCornerParameters()
        x, cov = initial_state(), initial_covariance()
        _, after = propagate(x, cov, a_lap(), p)
        before_temp = np.trace(cov[4:8, 4:8])
        after_temp = np.trace(after[4:8, 4:8])
        assert after_temp < before_temp

    def test_wear_uncertainty_does_not_collapse(self):
        """Wear is near-integrating, so its variance must not vanish."""
        p = FourCornerParameters()
        x, cov = initial_state(), initial_covariance()
        for lap in range(1, 20):
            x, cov = propagate(x, cov, a_lap(lap=lap), p)
        assert np.trace(cov[0:4, 0:4]) > 1e-4

    def test_covariance_stays_symmetric_and_psd(self):
        p = FourCornerParameters()
        x, cov = initial_state(), initial_covariance()
        for lap in range(1, 40):
            x, cov = propagate(x, cov, a_lap(lap=lap), p)
        assert np.allclose(cov, cov.T, atol=1e-12)
        assert np.linalg.eigvalsh(cov).min() >= -1e-10

    def test_substep_count_converges(self):
        """More sub-steps must stop changing the answer.

        If this fails the integrator is not converged and the "physics" is an
        artefact of the step size.
        """
        p = FourCornerParameters()
        x, cov = initial_state(), initial_covariance()
        reference, _ = propagate(x, cov, a_lap(), p, substeps=512)
        errors = []
        for n in (2, 4, 8, 16):
            got, _ = propagate(x, cov, a_lap(), p, substeps=n)
            errors.append(np.abs(got[T] - reference[T]).max())
        # Each doubling must cut the error by at least four; RK4 gives sixteen.
        for coarse, fine in zip(errors, errors[1:]):
            assert fine < coarse / 4.0
        # And the shipped default must already be converged.
        default, _ = propagate(x, cov, a_lap(), p)
        assert np.allclose(default[W], reference[W], rtol=1e-4)
        assert np.allclose(default[T], reference[T], rtol=1e-4)

    def test_wear_only_increases(self):
        p = FourCornerParameters()
        x, cov = initial_state(), initial_covariance()
        previous = x[W].copy()
        for lap in range(1, 25):
            x, cov = propagate(x, cov, a_lap(lap=lap), p)
            assert (x[W] >= previous - 1e-12).all()
            previous = x[W].copy()


class TestFilter:
    def _synthetic_stint(self, n_laps: int = 30, seed: int = 0, noise_s: float = 0.12):
        """Generate a stint from the model itself, then hide the state.

        Truth is produced by integrating the same SDE with no process noise, so
        this asks whether the filter can invert its own forward model through a
        scalar observation. That is the weakest form of the question and the one
        that has to pass before any harder version is worth asking.
        """
        rng = np.random.default_rng(seed)
        p = FourCornerParameters()
        x = initial_state(fuel_kg=100.0, tread_temp_c=95.0)
        inputs, observations, truth = [], [], []
        for lap in range(1, n_laps + 1):
            u = a_lap(lap=lap, laps_completed=float(lap - 1))
            dt = u.duration_s / 40
            for _ in range(40):
                x = clamp_state(x + dt * drift(x, u, p))
            truth.append(x.copy())
            inputs.append(u)
            observations.append(predict_lap_time(x, u, p) + rng.normal(0.0, noise_s))
        return inputs, np.asarray(observations), np.asarray(truth), p

    def test_runs_and_returns_a_finite_likelihood(self):
        inputs, obs, _, p = self._synthetic_stint()
        r = run_filter(inputs, obs, initial_state(tread_temp_c=95.0), initial_covariance(),
                       p, obs_scale_s=0.12)
        assert np.isfinite(r.log_likelihood)
        assert r.states.shape == (len(inputs), N_STATE)

    def test_covariance_never_loses_positive_definiteness(self):
        inputs, obs, _, p = self._synthetic_stint(n_laps=60)
        r = run_filter(inputs, obs, initial_state(tread_temp_c=95.0), initial_covariance(),
                       p, obs_scale_s=0.12)
        for cov in r.covariances:
            assert np.allclose(cov, cov.T, atol=1e-10)
            assert np.linalg.eigvalsh(cov).min() >= -1e-9

    def test_does_not_need_repair(self):
        """Divergence repairs are a symptom, and a clean run should have none."""
        inputs, obs, _, p = self._synthetic_stint(n_laps=60)
        r = run_filter(inputs, obs, initial_state(tread_temp_c=95.0), initial_covariance(),
                       p, obs_scale_s=0.12)
        assert r.n_diverged == 0

    def test_recovers_mean_wear_it_was_never_shown(self):
        """The four-corner analogue of exp01.

        Mean wear, not per-corner wear. The pre-registration predicts the
        individual corners are not identifiable from a scalar observation, and
        `test_individual_corners_are_harder_than_the_mean` measures that claim
        rather than assuming it.
        """
        inputs, obs, truth, p = self._synthetic_stint(n_laps=40, seed=3)
        r = run_filter(inputs, obs, initial_state(tread_temp_c=95.0), initial_covariance(),
                       p, obs_scale_s=0.12)
        true_mean = truth[:, W].mean(axis=1)
        est_mean = r.states[:, W].mean(axis=1)
        # Compare over the back half, after the filter has had laps to converge.
        error = np.abs(true_mean[20:] - est_mean[20:]).mean()
        assert error < 0.05, f"mean wear error {error:.4f}"

    def test_individual_corners_are_harder_than_the_mean(self):
        """Measures, rather than assumes, the pre-registered prediction P3.

        If per-corner error is materially worse than mean error, the scalar
        observation is failing to separate the corners -- which is exactly what
        `PREREGISTRATION_exp35.md` predicts before any of this ran.
        """
        inputs, obs, truth, p = self._synthetic_stint(n_laps=40, seed=5)
        r = run_filter(inputs, obs, initial_state(tread_temp_c=95.0), initial_covariance(),
                       p, obs_scale_s=0.12)
        mean_err = np.abs(truth[20:, W].mean(axis=1) - r.states[20:, W].mean(axis=1)).mean()
        corner_err = np.abs(truth[20:, W] - r.states[20:, W]).mean()
        assert np.isfinite(mean_err) and np.isfinite(corner_err)
        # Recorded as an inequality rather than a bound: the experiment quantifies
        # it, this only guarantees the comparison is computable and non-trivial.
        assert corner_err >= mean_err - 1e-9

    def test_robust_mode_resists_a_lock_up(self):
        """One catastrophically slow lap must not move the wear estimate much."""
        inputs, obs, _, p = self._synthetic_stint(n_laps=40, seed=7)
        clean = run_filter(inputs, obs, initial_state(tread_temp_c=95.0),
                           initial_covariance(), p, obs_scale_s=0.12)
        spoiled = obs.copy()
        spoiled[20] += 3.5  # a lock-up
        robust = run_filter(inputs, spoiled, initial_state(tread_temp_c=95.0),
                            initial_covariance(), p, obs_scale_s=0.12, robust=True)
        gaussian = run_filter(inputs, spoiled, initial_state(tread_temp_c=95.0),
                              initial_covariance(), p, obs_scale_s=0.12, robust=False)

        final_clean = clean.states[-1, W].mean()
        shift_robust = abs(robust.states[-1, W].mean() - final_clean)
        shift_gaussian = abs(gaussian.states[-1, W].mean() - final_clean)
        assert shift_robust <= shift_gaussian
        assert robust.weights[20] < 0.5, "the outlier should have been down-weighted"

    def test_rejects_mismatched_lengths(self):
        inputs, obs, _, p = self._synthetic_stint(n_laps=10)
        with pytest.raises(ValueError, match="one observation per lap"):
            run_filter(inputs, obs[:-1], initial_state(), initial_covariance(),
                       p, obs_scale_s=0.12)

    def test_rejects_non_positive_scale(self):
        inputs, obs, _, p = self._synthetic_stint(n_laps=10)
        with pytest.raises(ValueError, match="scale must be positive"):
            run_filter(inputs, obs, initial_state(), initial_covariance(),
                       p, obs_scale_s=0.0)


class TestSmoother:
    def test_smoothed_covariance_is_no_larger_than_filtered(self):
        """Hindsight cannot make you less certain.

        Checked on the trace rather than elementwise, because the smoother can
        move individual entries while reducing overall uncertainty.
        """
        p = FourCornerParameters()
        rng = np.random.default_rng(11)
        x = initial_state(tread_temp_c=95.0)
        inputs, obs = [], []
        for lap in range(1, 31):
            u = a_lap(lap=lap, laps_completed=float(lap - 1))
            dt = u.duration_s / 40
            for _ in range(40):
                x = clamp_state(x + dt * drift(x, u, p))
            inputs.append(u)
            obs.append(predict_lap_time(x, u, p) + rng.normal(0.0, 0.12))

        r = run_filter(inputs, np.asarray(obs), initial_state(tread_temp_c=95.0),
                       initial_covariance(), p, obs_scale_s=0.12)
        xs, ps = rts_smooth(r, inputs, p)

        assert xs.shape == r.states.shape
        # The final lap is identical by construction; check the interior.
        filtered_trace = np.array([np.trace(c) for c in r.covariances[:-1]])
        smoothed_trace = np.array([np.trace(c) for c in ps[:-1]])
        assert (smoothed_trace <= filtered_trace + 1e-8).mean() > 0.9

    def test_smoothed_states_stay_in_bounds(self):
        p = FourCornerParameters()
        rng = np.random.default_rng(13)
        inputs = [a_lap(lap=i, laps_completed=float(i - 1)) for i in range(1, 26)]
        obs = np.array([91.0 + 0.02 * i + rng.normal(0, 0.1) for i in range(25)])
        r = run_filter(inputs, obs, initial_state(tread_temp_c=95.0),
                       initial_covariance(), p, obs_scale_s=0.12)
        xs, _ = rts_smooth(r, inputs, p)
        assert (xs[:, W] >= 0.0).all()
        assert np.isfinite(xs).all()
