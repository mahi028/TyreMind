"""Download the open-access literature into `research/papers/`.

Only open-access sources are listed. arXiv, MDPI, PMC and university repositories
serve PDFs directly; SAE, Elsevier and Springer do not, and a paywalled paper is
recorded in `research/papers/PAYWALLED.md` with its DOI rather than fetched.

Re-running is cheap: a file that already exists is skipped, so this can be used to
top up the collection as new entries are added to CATALOGUE.

    python scripts/fetch_papers.py
    python scripts/fetch_papers.py --only 02_tyre_wear_physics
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("research/papers")
UA = "TyreMind-research/0.1 (academic use; contact via repository)"

#: (folder, filename stem, url, citation). The stem carries first author and year
#: so the folder is readable without opening anything.
CATALOGUE: list[tuple[str, str, str, str]] = [
    # ---- 01 race strategy and race simulation ----
    ("01_race_strategy_simulation", "Heilmeier2020_VirtualStrategyEngineer",
     "https://www.mdpi.com/2076-3417/10/21/7805/pdf",
     "Heilmeier, Thomaser, Graf, Betz, Lienkamp (2020). Virtual Strategy Engineer: "
     "Using Artificial Neural Networks for Making Race Strategy Decisions in Circuit "
     "Motorsport. Applied Sciences 10(21):7805."),
    ("01_race_strategy_simulation", "Heilmeier2020_MonteCarloRaceSimulation",
     "https://www.mdpi.com/2076-3417/10/12/4229/pdf",
     "Heilmeier, Graf, Betz, Lienkamp (2020). Application of Monte Carlo Methods to "
     "Consider Probabilistic Effects in a Race Simulation for Circuit Motorsport. "
     "Applied Sciences 10(12):4229."),
    ("01_race_strategy_simulation", "Bassi2023_MasteringNordschleife",
     "https://arxiv.org/pdf/2306.16088",
     "Mastering Nordschleife: A comprehensive race simulation for AI strategy "
     "decision-making in motorsports. arXiv:2306.16088."),
    ("01_race_strategy_simulation", "Pitwall_CalibratedMonteCarloBriefings",
     "https://arxiv.org/pdf/2607.06495",
     "Pitwall: Faithful Natural-Language Race-Strategy Briefings from a Calibrated "
     "Real-Time Monte Carlo Engine. arXiv:2607.06495."),
    ("01_race_strategy_simulation", "TowardsLearningBasedF1RaceStrategies",
     "https://arxiv.org/pdf/2512.21570",
     "Towards Learning-Based Formula 1 Race Strategies. arXiv:2512.21570."),

    # ---- 02 tyre wear physics ----
    ("02_tyre_wear_physics", "RubberWear_HistoryMechanismsPerspectives",
     "https://arxiv.org/pdf/2503.19494",
     "Rubber Wear: History, Mechanisms, and Perspectives. arXiv:2503.19494."),
    ("02_tyre_wear_physics", "Farroni2023_TireWearSensitivityAnalysis",
     "https://www.megaride.eu/wp-content/uploads/2023/07/lubricants-11-00269.pdf",
     "Tire Wear Sensitivity Analysis and Modeling Based on a Statistical Multidisciplinary "
     "Approach for High-Performance Vehicles. Lubricants 11(6):269."),
    ("02_tyre_wear_physics", "Wear_IrreversibleEntropyGeneration",
     "https://arxiv.org/pdf/1008.0412",
     "The Relation Between Wear and Irreversible Entropy Generation in the Dry Sliding "
     "of Metals. arXiv:1008.0412."),

    # ---- 03 tyre friction and grip ----
    ("03_tyre_friction_grip", "StringTyreModels_DistributedFrBD",
     "https://arxiv.org/pdf/2603.02869",
     "Bare and stretched string tyre models with distributed FrBD dynamics. "
     "arXiv:2603.02869."),

    # ---- 05 lap time modelling ----
    ("05_lap_time_modelling", "RealTimeOptimalTrajectory_LapTimeML",
     "https://arxiv.org/pdf/2102.02315",
     "Real-Time Optimal Trajectory Planning for Autonomous Vehicles and Lap Time "
     "Simulation Using Machine Learning. arXiv:2102.02315."),

    # ---- 06 ML degradation ----
    ("06_ml_degradation", "Sulsters2025_ExplainableTyreEnergyF1",
     "https://arxiv.org/pdf/2501.04067",
     "Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy. "
     "arXiv:2501.04067 / ACM SAC 2025."),

    # ---- 03 tyre friction and grip ----
    ("03_tyre_friction_grip", "Farroni2019_TRT_EVO_ThermodynamicTireModel",
     "https://www.megaride.eu/wp-content/uploads/2022/06/0954407018808992.pdf",
     "Farroni, Russo, Sakhnevych, Timpone (2019). TRT EVO: Advances in real-time "
     "thermodynamic tire modeling for vehicle dynamics simulations. Proc IMechE Part D."),
    ("04_thermal_models", "TireThermalCharacterization_TestProcedureModel",
     "https://www.iaeng.org/publication/WCE2016/WCE2016_pp1199-1204.pdf",
     "Tire Thermal Characterization: Test Procedure and Model Parameters Evaluation. "
     "Proc. World Congress on Engineering 2016, pp.1199-1204."),

    # ---- per-corner load transfer: the four-tyre question ----
    ("03_tyre_friction_grip", "LoadTransferMitigation_MPC",
     "https://arxiv.org/pdf/2606.26313",
     "Racing a Wheeled Quadruped: Active Load Transfer Mitigation via Model Predictive "
     "Control. arXiv:2606.26313.  Load-transfer ratio formulation."),
    ("03_tyre_friction_grip", "GM3_GeneralPhysicalModel",
     "https://arxiv.org/pdf/2510.07807",
     "GM3: A General Physical Model for Micro-Mobility Vehicles. arXiv:2510.07807.  "
     "Longitudinal and lateral load transfer with CoG height, wheelbase, track width."),
    ("03_tyre_friction_grip", "AutoRally_DoubleTrackModel",
     "https://arxiv.org/pdf/1806.00678",
     "AutoRally: An Open Platform for Aggressive Autonomous Driving. arXiv:1806.00678.  "
     "Double-track model with per-side load difference."),
    ("03_tyre_friction_grip", "MultichamberSuspension_LoadTransfer",
     "https://arxiv.org/pdf/2304.08201",
     "Handling-Oriented Stiffness Control of a Multichamber Suspension. arXiv:2304.08201."),

    # ---- 09 conformal prediction: the uncertainty layer we ship ----
    ("09_conformal_uncertainty", "GibbsCandes2021_AdaptiveConformalInference",
     "https://arxiv.org/pdf/2106.00170",
     "Gibbs & Candes (2021). Adaptive Conformal Inference Under Distribution Shift. "
     "NeurIPS 2021.  *** THE ACI UPDATE RULE WE IMPLEMENT ***"),
    ("09_conformal_uncertainty", "AngelopoulosBates_GentleIntroConformal",
     "https://arxiv.org/pdf/2107.07511",
     "Angelopoulos & Bates. A Gentle Introduction to Conformal Prediction and "
     "Distribution-Free Uncertainty Quantification. arXiv:2107.07511."),
    ("09_conformal_uncertainty", "Lei2016_DistributionFreePredictiveInference",
     "https://arxiv.org/pdf/1604.04173",
     "Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018). Distribution-Free Predictive "
     "Inference for Regression. JASA.  *** SPLIT CONFORMAL, THE FINITE-SAMPLE QUANTILE ***"),
    ("09_conformal_uncertainty", "Tibshirani2019_ConformalCovariateShift",
     "https://arxiv.org/pdf/1904.06019",
     "Tibshirani, Barber, Candes, Ramdas (2019). Conformal Prediction Under Covariate "
     "Shift. NeurIPS 2019."),

    # ---- 11 pit stop prediction: the direct benchmark for exp22 ----
    ("11_pit_stop_prediction", "Frontiers2025_PitStopDecisionSupport_DeepLearning",
     "https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/pdf",
     "Data-driven pit stop decision support for Formula 1 using deep learning models. "
     "Frontiers in Artificial Intelligence (2025), doi:10.3389/frai.2025.1673148.  "
     "*** Bi-LSTM precision 0.77, recall 0.86, F1 0.81 on 3% positive class. "
     "THE NUMBER exp22 MUST BEAT ***"),
    ("11_pit_stop_prediction", "NCIRL_PitstopStrategyEnsembleLearning",
     "https://norma.ncirl.ie/7601/1/anikethmaheshrao.pdf",
     "Predictive Model for Pitstop Strategy in Formula 1 using Ensemble Learning. "
     "National College of Ireland thesis."),
    ("11_pit_stop_prediction", "Preprints2025_MachineLearningPredictingF1",
     "https://www.preprints.org/manuscript/202504.1471/v1/download",
     "The Use of Machine Learning in Predicting Formula 1 Results. Preprints 2025."),

    # ---- 07 prognostics / RUL: the cross-domain evidence base ----
    ("07_prognostics_rul", "UncertaintyAwareRUL_TurbofanAleatoric",
     "https://arxiv.org/pdf/2511.19124",
     "Uncertainty-Aware Deep Learning Framework for RUL Prediction in Turbofan Engines "
     "with Learned Aleatoric Uncertainty. arXiv:2511.19124.  Reports 93.5-95.2% coverage "
     "on 95% intervals -- the direct comparator for our conformal RUL."),
    ("07_prognostics_rul", "BenchmarkUQ_DeepLearningPrognostics",
     "https://arxiv.org/pdf/2302.04730",
     "A Benchmark on Uncertainty Quantification for Deep Learning Prognostics. "
     "arXiv:2302.04730."),
    ("07_prognostics_rul", "UQ_Tutorial_EngineeringDesignHealthPrognostics",
     "https://arxiv.org/pdf/2305.04933",
     "Uncertainty Quantification in Machine Learning for Engineering Design and Health "
     "Prognostics: A Tutorial. arXiv:2305.04933."),
    ("07_prognostics_rul", "RUL_UQ_NonstationaryGaussianProcess",
     "https://arxiv.org/pdf/2109.12111",
     "Accurate RUL Prediction with Uncertainty Quantification: a Deep Learning and "
     "Nonstationary Gaussian Process Approach. arXiv:2109.12111."),
    ("07_prognostics_rul", "BayesianDL_RUL_SteinVariational",
     "https://arxiv.org/pdf/2402.01098",
     "Bayesian Deep Learning for RUL Estimation via Stein Variational Gradient Descent. "
     "arXiv:2402.01098."),

    # ---- 10 theses and reports ----
    ("10_theses_reports", "Sulsters_SimulatingF1RaceStrategies",
     "https://vu-business-analytics.github.io/internship-office/papers/paper-sulsters.pdf",
     "Sulsters, C. Simulating Formula One Race Strategies. VU Amsterdam Business "
     "Analytics internship paper."),
    ("10_theses_reports", "UChile_OptimizationPitStopStrategiesF1",
     "https://repositorio.uchile.cl/bitstream/handle/2250/199664/Optimization-of-pit-stop-strategies-in-Formula-1-racing.pdf?sequence=1&isAllowed=y",
     "Optimization of Pit Stop Strategies in Formula 1 Racing. Universidad de Chile."),
    ("10_theses_reports", "PitStopStrategyOptimizationModel_SimulationFramework",
     "https://optimization-online.org/wp-content/uploads/2026/02/Pit_Stop_Strategy_Optimization_Model.pdf",
     "A simulation framework for Formula 1 race strategy. Optimization Online (2026)."),

    # ---- 08 state-space and filtering ----
    ("08_state_space_filtering", "StateSpaceApproach_TireDegradationF1",
     "https://arxiv.org/pdf/2512.00640",
     "A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing. "
     "arXiv:2512.00640.  *** CLOSEST PRIOR ART TO TYREMIND -- READ FIRST ***"),
]


def fetch(url: str, destination: Path, *, timeout: int = 90) -> tuple[bool, str]:
    """Fetch one PDF. Returns (ok, message) rather than raising.

    One unreachable publisher must not stop the rest of the download, so every
    failure is reported and recorded instead of propagating.
    """
    if destination.exists() and destination.stat().st_size > 20_000:
        return True, f"skip (have {destination.stat().st_size // 1024} KB)"

    request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return False, f"FAILED {exc}"

    # Publishers sometimes answer a PDF request with an HTML interstitial and a
    # 200, which would otherwise be saved as a corrupt .pdf nobody notices.
    if not payload.startswith(b"%PDF"):
        return False, f"FAILED not a PDF ({len(payload)} bytes, starts {payload[:20]!r})"

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return True, f"ok {len(payload) // 1024} KB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="restrict to one folder")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    index: list[dict] = []
    failures: list[dict] = []

    for folder, stem, url, citation in CATALOGUE:
        if args.only and folder != args.only:
            continue
        destination = ROOT / folder / f"{stem}.pdf"
        ok, message = fetch(url, destination)
        print(f"  {folder}/{stem}: {message}")
        record = {"folder": folder, "stem": stem, "url": url, "citation": citation,
                  "status": message}
        (index if ok else failures).append(record)
        if not message.startswith("skip"):
            time.sleep(args.delay)

    ROOT.joinpath("INDEX.json").write_text(
        json.dumps({"downloaded": index, "failed": failures}, indent=2), encoding="utf-8")
    print(f"\n{len(index)} available, {len(failures)} failed -> {ROOT / 'INDEX.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
