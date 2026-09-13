import numpy as np, torch, yaml
from src.sim.simulator import SimulatorConfig, generate_synthetic_dataset
from src.data.features import split_by_event
from src.models.dataset import Vocabs
from src.models.nam import NAM, NAMConfig
from src.train import TrainConfig, fit, pick_device
from src.evaluate import detect_cliff, recovered_slope

cfg = SimulatorConfig.from_yaml()
raw_yaml = yaml.safe_load(open("config/default.yaml"))
device = pick_device()

df_main, truth = generate_synthetic_dataset(cfg, seed=42, return_truth=True)
train_main, val_main = split_by_event(df_main, seed=42)

df_neg = generate_synthetic_dataset(cfg, seed=7, zero_degradation=True)
train_neg, val_neg = split_by_event(df_neg, seed=7)


def run(smooth_w):
    train_cfg = TrainConfig.from_yaml_dict(raw_yaml)
    train_cfg.tyre_smoothness_weight = smooth_w

    vocabs = Vocabs.fit(train_main)
    nam_cfg = NAMConfig(n_circuits=vocabs.circuit.size, n_entries=vocabs.entry.size,
                         n_compounds=vocabs.compound.size, max_age=cfg.max_age,
                         fuel_time_per_kg_s=cfg.fuel_time_per_kg_s)
    model = NAM(nam_cfg)
    fit(model, train_main, val_main, vocabs, train_cfg, device=device, verbose=False)
    model.eval()

    slope_diffs = {}
    cliff_diffs = {}
    circuit_id = next(iter(vocabs.circuit.mapping.values()))
    for compound in cfg.compounds:
        cid = vocabs.compound.mapping[compound]
        curve = model.degradation_curve(compound_id=cid, circuit_id=circuit_id, track_temp_c=35.0, device=device).cpu().numpy()
        rec_slope = recovered_slope(curve)
        true_slope = cfg.true_linear_slope_s_per_lap[compound]
        slope_diffs[compound] = abs(rec_slope - true_slope)
        detected = detect_cliff(curve)
        true_cliff = cfg.true_cliff_age[compound]
        cliff_diffs[compound] = None if detected is None else abs(detected - true_cliff)

    vocabs_neg = Vocabs.fit(train_neg)
    nam_cfg_neg = NAMConfig(n_circuits=vocabs_neg.circuit.size, n_entries=vocabs_neg.entry.size,
                             n_compounds=vocabs_neg.compound.size, max_age=cfg.max_age,
                             fuel_time_per_kg_s=cfg.fuel_time_per_kg_s)
    model_neg = NAM(nam_cfg_neg)
    fit(model_neg, train_neg, val_neg, vocabs_neg, train_cfg, device=device, verbose=False)
    model_neg.eval()
    circuit_id_neg = next(iter(vocabs_neg.circuit.mapping.values()))
    neg_slopes = {}
    for compound in cfg.compounds:
        cid = vocabs_neg.compound.mapping[compound]
        curve = model_neg.degradation_curve(compound_id=cid, circuit_id=circuit_id_neg, track_temp_c=35.0, device=device).cpu().numpy()
        neg_slopes[compound] = recovered_slope(curve)

    print(f"smooth_w={smooth_w}")
    print(f"  slope_diffs={ {k: round(v,4) for k,v in slope_diffs.items()} } max={max(slope_diffs.values()):.4f}")
    print(f"  cliff_diffs={cliff_diffs}")
    print(f"  neg_control_slopes={ {k: round(v,4) for k,v in neg_slopes.items()} } max={max(neg_slopes.values()):.4f}")


for w in [0.2, 0.4, 0.6]:
    run(w)
