from __future__ import annotations

import pandapower.networks as pn
import argparse
from src.pipeline.run_pipeline import (run_pipeline, PipelineConfig, ScenarioConfig )
from src.pipeline.attack_schedule import generate_random_attack
import numpy as np
from pathlib import Path
import secrets
from src.io.export_pipeline_run import export_pipeline_run


def build_ieee9_network():
    # Build and return the IEEE 9-bus test network
    return pn.case9()

DEFAULT_RANDOM_P_START = 0.03
DEFAULT_RANDOM_DUR_MIN = 5
DEFAULT_RANDOM_DUR_MAX = 40
DEFAULT_RANDOM_COOLDOWN = 10
DEFAULT_RANDOM_NO_ATTACK_BEFORE = 200

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

    parser.add_argument(
    "--episodes",
    type=str,
    default=None,
    help=(
        "Optional attack episodes as a string, e.g. "
        "'50:150' or '50:150,300:360'. "
        "If provided, overrides --attack_start/--attack_end."
        ),
    )
    parser.add_argument(
    "--episode_seed",
    type=int,
    default=None,
    help="Seed for attack episode generation. If omitted, a random seed is used and logged.",
)

    parser.add_argument("--random_episodes", action="store_true")
    parser.add_argument("--p_start", type=float, default=DEFAULT_RANDOM_P_START)
    parser.add_argument("--duration_min", type=int, default=DEFAULT_RANDOM_DUR_MIN)
    parser.add_argument("--duration_max", type=int, default=DEFAULT_RANDOM_DUR_MAX)
    parser.add_argument("--cooldown", type=int, default=DEFAULT_RANDOM_COOLDOWN)
    parser.add_argument("--no_attack_before", type=int, default=DEFAULT_RANDOM_NO_ATTACK_BEFORE)


    # Attack indice parsers
    parser.add_argument("--random_strength", action="store_true")
    parser.add_argument("--strength_min", type=float, default=0.5)
    parser.add_argument("--strength_max", type=float, default=1.5)
    parser.add_argument(
        "--attack_schedule",
        type=str,
        choices=["fixed", "random"],
        default="fixed",
        help="Attack schedule mode: fixed window or random episodic",
    )

    return parser.parse_args()

def parse_episodes(episodes_str: str | None):
    """
    Parse episodes from a string like:
      '50:150' or '50:150,300:360'
    into [(50,150), (300,360)]
    """
    if episodes_str is None:
        return None

    episodes = []
    for block in episodes_str.split(","):
        try:
            s, e = block.split(":")
            episodes.append((int(s), int(e)))
        except ValueError:
            raise ValueError(
                f"Invalid episode specification '{block}'. "
                "Expected format 'start:end'."
            )
    return episodes

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
    # Resolve episode seed
    if args.episode_seed is None:
        episode_seed = secrets.randbits(32)
        print(f"[INFO] No episode_seed provided. Generated random episode_seed = {episode_seed}")
    else:
        episode_seed = args.episode_seed
        print(f"[INFO] Using provided episode_seed = {episode_seed}")

    episode_rng = np.random.default_rng(episode_seed)

    episodes = None
    if args.attack_schedule == "random":
        # episode_rng = np.random.default_rng(episode_seed)

        episodes = generate_random_attack(
            T=args.T,
            rng=episode_rng,
            p_start=args.p_start,
            duration_min=args.duration_min,
            duration_max=args.duration_max,
            cooldown=args.cooldown,
            no_attack_before=args.no_attack_before,
        )

        print("[INFO] Using RANDOM episodic schedule")
        print("Generated attack episodes:", episodes)
    else:
        print("[INFO] Using FIXED attack window")

    scenario_config = ScenarioConfig(
        attack_type=args.scenario,
        start=args.attack_start,
        end=args.attack_end,
        episodes=episodes,
        episode_seed=episode_seed,
        random_strength=args.random_strength,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
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
        / f"T_{pipeline_config.T}"
        / f"seed_{pipeline_config.seed}"
    )
    export_pipeline_run(outputs, run_dir)

    # Print execution summary
    print("\n=== IEEE-9 Pipeline Run Complete ===")
    print(f"Scenario        : {scenario_config.attack_type}")
    print(f"Seed            : {pipeline_config.seed}")
    print(f"Timesteps (T)   : {pipeline_config.T}")
    if scenario_config.episodes is not None:
        print(f"Schedule        : random ")
        print(f"Episodes        : {scenario_config.episodes})")
        print(f"Episode seed    : {episode_seed}")
    else:
        print(f"Schedule        : fixed ")
        print(f"Attack window   : [{scenario_config.start}, {scenario_config.end})")
    print(f"Measurements   : {outputs.Z_clean.shape[1]}")
    print(f"State dim       : {outputs.X_hat.shape[1]}")
    print(f"Attacked steps  : {int(outputs.attack_mask.sum())}")
    print(f"Converged steps : {int(outputs.converged.sum())}")
    print("===================================\n")
    
if __name__ == "__main__":
    print("Running IEEE-9 pipeline...")
    main()
