from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from src.pipeline.run_pipeline import PipelineOutputs

def export_pipeline_run(
    outputs: PipelineOutputs,
    out_dir: Path,

) -> None:
    
    # Minimal export of pipeline ouputs.

    out_dir.mkdir(parents=True, exist_ok=True)
    T = outputs.time.shape[0]

    # Measurements 
    meas_df = pd.DataFrame(
        outputs.Z_attacked,
        columns=[f"z_{i}" for i in range(outputs.Z_attacked.shape[1])],
    )
    meas_df.insert(0, "t", outputs.time)
    meas_df.to_csv(out_dir / "measurements.csv", index=False)

    # Residual norms
    resid_df = pd.DataFrame({
        "t": outputs.time,
        "residual_norm": outputs.residual_norms,
    })
    resid_df.to_csv(out_dir / "residuals.csv", index=False)

    # Attack mask
    attack_df = pd.DataFrame({
        "t": outputs.time,
        "attack_mask": outputs.attack_mask.astype(int),
    })
    attack_df.to_csv(out_dir / "attack_mask.csv", index=False)

    # Convergence Mask
    converged_df = pd.DataFrame({
        "t": outputs.time,
        "converged": outputs.converged.astype(int),
    })
    converged_df.to_csv(out_dir / "convergence_mask.csv", index=False)

    # Metadata
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(outputs.metadata, f, indent=2)