from __future__ import annotations

import pandapower.networks as pn
import argparse
from src.pipeline.run_pipeline import (run_pipeline, PipelineConfig, ScenarioConfig )

from pathlib import Path
from src.io.export_pipeline_run import export_pipeline_run

def build_ieee9_network():
    # Build and return the IEEE 9-bus test network
    return pn.case9()

def parse_args():
    parser = argparse.ArgumentParser(description="Run IEEE-9 FDIA pipeline")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["standard", "random", "stealth"],
        default="standard",
        help="Type of FDIA scenario to run",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--attack_start", type=int, default=50)
    parser.add_argument("--attack_end", type=int, default=150)

    return parser.parse_args()

def main():
    args = parse_args()

    # Build network
    net = build_ieee9_network()

    # Construct configs
    pipeline_config = PipelineConfig(
        network="ieee9",
        seed=args.seed,
        T=args.T,
    )

    scenario_config = ScenarioConfig(
        attack_type=args.scenario,
        start=args.attack_start,
        end=args.attack_end,
    )
    
    # Run pipeline
    outputs = run_pipeline(
        net,
        config=pipeline_config,
        scenario=scenario_config,
    )

    # Decide output location
    run_dir = (
        Path("runs")
        / "ieee9"
        / scenario_config.attack_type
        / f"seed_{pipeline_config.seed}"
    )
    export_pipeline_run(outputs, run_dir)

    # Print execution summary
    print("\n=== IEEE-9 Pipeline Run Complete ===")
    print(f"Scenario        : {scenario_config.attack_type}")
    print(f"Seed            : {pipeline_config.seed}")
    print(f"Timesteps (T)   : {pipeline_config.T}")
    print(f"Attack window   : [{scenario_config.start}, {scenario_config.end})")
    print(f"Measurements   : {outputs.Z_clean.shape[1]}")
    print(f"State dim       : {outputs.X_hat.shape[1]}")
    print(f"Attacked steps  : {int(outputs.attack_mask.sum())}")
    print(f"Converged steps : {int(outputs.converged.sum())}")
    print("===================================\n")

if __name__ == "__main__":
    print("Running IEEE-9 pipeline...")
    main()
