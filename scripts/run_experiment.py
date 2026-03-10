import subprocess
from pathlib import Path
import datetime
import shutil

BASE_ARGS = [
    "python", "-m", "scripts.run_live_ieee9",
    "--scenario", "stealth",
    "--attack_schedule", "fixed",
    "--attack_start", "200",
    "--attack_end", "260",
    "--attack_strength", "0.15",
    "--attack_bus", "4",
    "--detector_type", "ocsvm",
    "--representation", "residuals",
    "--window_size", "15",
    "--detector_dir", "trained_detectors_streaming_final",
    "--stop_after_steps", "600"
]

OUTPUT_ROOT = Path("runs_final/stealth_experiments")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def run_variant(name, extra_flags):
    print(f"\n===== Running {name} =====")

    cmd = BASE_ARGS + extra_flags
    subprocess.run(cmd, check=True)

    # Find newest run folder
    runs_dir = Path("runs_live/ieee9/stealth")
    latest_run = max(runs_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)

    dest = OUTPUT_ROOT / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(latest_run, dest)

    print(f"Saved to: {dest}")


if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n=== FINAL STEALTH EXPERIMENT SUITE ({timestamp}) ===")

    # A — Detection only
    run_variant(
        "A_detection_only",
        []
    )

    # B — Detection + Mitigation
    run_variant(
        "B_detection_mitigation",
        ["--enable_mitigation"]
    )

    # C — Detection + Mitigation + Control
    run_variant(
        "C_detection_mitigation_control",
        [
            "--enable_mitigation",
            "--enable_control",
            "--control_on_alarm"
        ]
    )

    # D — Full system (Recovery auto-enabled internally)
    run_variant(
        "D_full_system_recovery",
        [
            "--enable_mitigation",
            "--enable_control",
            "--control_on_alarm"
        ]
    )

    print("\nAll final experiments completed.")
