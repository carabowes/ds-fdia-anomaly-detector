from __future__ import annotations

import argparse
from pathlib import Path
import pandapower.networks as pn
import pandapower as pp
import pickle
import numpy as np

from src.pipeline.run_pipeline import PipelineConfig, ScenarioConfig
from src.pipeline.streaming import run_streaming_pipeline
from src.control.opf_controller import OPFController
from src.pipeline.simulation import build_dc_measurement_model
from src.pipeline.attack_targets import choose_attack_buses_ieee9



import warnings
warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning
)

def build_ieee9_network():
    return pn.case9()

DEFAULT_RANDOM_P_START = 0.03
DEFAULT_RANDOM_DUR_MIN = 5
DEFAULT_RANDOM_DUR_MAX = 40
DEFAULT_RANDOM_COOLDOWN = 10
DEFAULT_RANDOM_NO_ATTACK_BEFORE = 200


def parse_args():
    parser = argparse.ArgumentParser(description="Run IEEE-9 LIVE streaming FDIA pipeline")

    parser.add_argument("--scenario", choices=["standard", "random", "stealth"], default="standard")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--attack_schedule", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--attack_start", type=int, default=50)
    parser.add_argument("--attack_end", type=int, default=150)

    parser.add_argument("--p_start", type=float, default=DEFAULT_RANDOM_P_START)
    parser.add_argument("--duration_min", type=int, default=DEFAULT_RANDOM_DUR_MIN)
    parser.add_argument("--duration_max", type=int, default=DEFAULT_RANDOM_DUR_MAX)
    parser.add_argument("--cooldown", type=int, default=DEFAULT_RANDOM_COOLDOWN)
    parser.add_argument("--no_attack_before", type=int, default=DEFAULT_RANDOM_NO_ATTACK_BEFORE)

    parser.add_argument("--representation", choices=["residuals", "innovations"], default="innovations")
    parser.add_argument("--innovation_alpha", type=float, default=0.3)
    parser.add_argument("--window_size", type=int, default=5)

    parser.add_argument("--out_root", type=str, default="runs_live")
    parser.add_argument("--stop_after_steps", type=int, default=None)

    parser.add_argument(
        "--detector_type",
        choices=["isolation_forest", "ocsvm", "lof", "none"],
        default="ocsvm",
    )

    parser.add_argument("--log_features", action="store_true")
    parser.add_argument(
        "--scaler_path",
        type=str,
        default=None,
        help="Path to trained StandardScaler (required for streaming detectors)",

    )
    parser.add_argument("--enable_control", action="store_true")
    parser.add_argument("--control_on_alarm", action="store_true")

    parser.add_argument(
        "--attack_buses",
        type=int,
        nargs="+",
        default=None,
        help="List of bus indices to attack (e.g. --attack_buses 4 5 7)",
    )


    parser.add_argument("--attack_strength", type=float, default=0.1, help="Stealth attack magnitude as fraction of measurement (e.g. 0.10 = 10%)")
    # parser.add_argument("--stealth_max_abs", type=float, default=0.02)
    # parser.add_argument("--stealth_jitter_std", type=float, default=0.002)
    # parser.add_argument(
    #     "--enable_mitigation",
    #     action="store_true",
    #     help="Enable measurement replacement mitigation on alarm",
    # )      

    parser.add_argument("--enable_mitigation", action="store_true")
    parser.add_argument("--mitigation_mode", type=str, default="freeze", choices=["freeze"])
    # parser.add_argument("--calibration_steps", type=int, default=None)


    return parser.parse_args()

def measurement_indices_for_bus(net, H, bus_idx):
    """
    Return indices in z corresponding to line flow measurements incident to the attacked bus.
    """
    indices = []

    for idx, line in net.line.iterrows():
        if line.from_bus == bus_idx or line.to_bus == bus_idx:
            indices.append(idx)
    return np.array(indices, dtype=int)


def main():
    args = parse_args()

    detector = None
    scaler = None
    controller = None
    attacked_indices = None

    net = build_ieee9_network()
    
    pp.runpp(net, algorithm="nr", init="dc", calculate_voltage_angles=True)

    # if args.enable_control:
    #     # Ramp limits in MW per step for each generator index in net.gen
    #     ramp_limits = {int(i): 10.0 for i in net.gen.index}
    #     controller = OPFController(ramp_limits=ramp_limits)
    #     print(f"[LIVE] Control enabled (ramp_limits={ramp_limits})")
    
    if args.enable_control:
        ramp_limits = {int(i): 10.0 for i in net.gen.index}

        control_bus = args.attack_buses[0] if args.attack_buses else None

        controller = OPFController(
            ramp_limits=ramp_limits,
            attack_bus=control_bus,
            gain=20.0,
        )

        print(
            f"[LIVE] Control enabled (ramp_limits={ramp_limits}, "
            f"attack_buses={args.attack_buses}, gain=20.0)"
        )
    # ---- Load streaming-trained detector ----
    if args.detector_type != "none":
        # detector_path = {
        #     "isolation_forest": f"trained_detectors_streaming/iforest_ieee9_W{args.window_size}.pkl",
        #     "ocsvm": f"trained_detectors_streaming/ocsvm_ieee9_W{args.window_size}.pkl",
        #     "lof": f"trained_detectors_streaming/lof_ieee9_W{args.window_size}.pkl",
        # }[args.detector_type]
        detector_path = {
            "isolation_forest": f"trained_detectors_streaming/iforest_ieee9_W{args.window_size}_{args.representation}.pkl",
            "ocsvm": f"trained_detectors_streaming/ocsvm_ieee9_W{args.window_size}_{args.representation}.pkl",
            "lof": f"trained_detectors_streaming/lof_ieee9_W{args.window_size}_{args.representation}.pkl",
        }[args.detector_type]


        with open(detector_path, "rb") as f:
            detector = pickle.load(f)

        print(f"[LIVE] Loaded detector: {detector_path}")
    else:
        print("[LIVE] Detector disabled")

    if detector is not None:
        if args.scaler_path is None:
            raise RuntimeError("Streaming detectors REQUIRE a scaler. Pass --scaler_path")
        with open(args.scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"[LIVE] Loaded scaler: {args.scaler_path}")

    # ---------------- ATTACK BUS SELECTION (STEALTH ONLY) ----------------
    attacked_indices = None  # default: no targeted subset (=> attack all in step_streaming)
    if args.scenario == "stealth":

        # ---- Choose buses ----
        if args.attack_buses is not None:
            attack_buses = [int(b) for b in args.attack_buses]
        else:
            # fallback to informed attacker logic
            targets = choose_attack_buses_ieee9(net)
            attack_buses = [targets["central_bus"]]

        # ---- Safety checks ----
        slack_bus = 0
        for b in attack_buses:
            if b < 0 or b >= len(net.bus):
                raise ValueError(f"attack_bus {b} out of range")
            if b == slack_bus:
                raise ValueError("Cannot attack slack bus (no state variable).")

        # ---- Build attacked measurement indices for ALL buses ----
        H_tmp, _, _, _ = build_dc_measurement_model(net)

        attacked_rows = []

        for bus in attack_buses:
            state_idx = bus - 1  # slack removed
            rows = np.where(np.abs(H_tmp[:, state_idx]) > 1e-6)[0]
            attacked_rows.extend(rows.tolist())

        attacked_indices = np.unique(attacked_rows).astype(int)

        print(
            f"[LIVE] Stealth attacker targeting buses {attack_buses} "
            f"({len(attacked_indices)} total measurement indices)"
        )

    # if args.scenario == "stealth":
    #     # Pick the bus (explicit CLI override OR informed attacker)
    #     if args.attack_bus is not None:
    #         attack_bus = int(args.attack_bus)
    #     else:
    #         targets = choose_attack_buses_ieee9(net)
    #         attack_bus = targets["central_bus"]

    #     # ---- Safety checks ----
    #     if attack_bus < 0 or attack_bus >= len(net.bus):
    #         raise ValueError(f"attack_bus must be in [0, {len(net.bus)-1}] for IEEE-9")

    #     slack_bus = 0
    #     if attack_bus == slack_bus:
    #         raise ValueError("Cannot attack slack bus (no state variable).")

    #     # ---- Build attacked measurement indices for THIS bus ----
    #     H_tmp, _, _, _ = build_dc_measurement_model(net)

    #     state_idx = attack_bus - 1  # slack removed
    #     rows = np.where(np.abs(H_tmp[:, state_idx]) > 1e-6)[0]

    #     if len(rows) == 0:
    #         raise RuntimeError(f"No valid measurement indices for bus {attack_bus}")

    #     attacked_indices = rows.astype(int)

    #     print(
    #         f"[LIVE] Stealth attacker targeting bus {attack_bus} "
    #         f"({len(attacked_indices)} measurement indices)"
    #     )



    config = PipelineConfig(
        network="ieee9",
        seed=args.seed,
        T=0,  # unused in streaming
    )

    scenario = ScenarioConfig(
        attack_type=args.scenario,
        start=args.attack_start,
        end=args.attack_end,
        episodes=None,
        episode_seed=None,
        attacked_indices=attacked_indices,

    )

    run_dir = run_streaming_pipeline(
        net,
        config=config,
        scenario=scenario,
        out_root=Path(args.out_root),
        detector=detector,
        scaler=scaler,
        window_size=args.window_size,
        representation=args.representation,
        innovation_alpha=args.innovation_alpha,
        attack_schedule_mode=args.attack_schedule,
        p_start=args.p_start,
        duration_min=args.duration_min,
        duration_max=args.duration_max,
        cooldown=args.cooldown,
        no_attack_before=args.no_attack_before,
        stop_after_steps=args.stop_after_steps,
        log_features=args.log_features,
        controller=controller, 
        control_on_alarm=args.control_on_alarm,
        attack_strength=args.attack_strength,
        enable_mitigation=args.enable_mitigation,
        mitigation_mode=args.mitigation_mode,
        # calibration_steps=args.calibration_steps,
    )

    print(f"[LIVE DONE] wrote: {run_dir}")


if __name__ == "__main__":
    main()
