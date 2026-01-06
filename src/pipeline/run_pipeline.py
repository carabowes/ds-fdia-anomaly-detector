from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from src.pipeline.time_series import run_time_series, inject_fdi_time_series
from src.pipeline.state_estimation import run_wls_time_series

ScenarioType = Literal["standard", "random", "stealth"]

@dataclass(frozen=True)
class ScenarioConfig:
    # Configuartion for a single scenario. Attacks are injected over a time window [start, end)

    attack_type: ScenarioType = "standard"
    attacked_indices: Optional[np.ndarray] = None
    start: int = 50
    end: int = 150

    # standard FDIA
    shift: float = 0.1
    # random FDIA
    scale: float = 0.05
    # stealth FDIA
    alpha: float = 0.05

@dataclass(frozen=True)
class PipelineConfig:
    network: Literal["ieee9"] = "ieee9"
    seed: int = 42

    # Time-series simulation
    T: int = 200
    p_noise_std: float = 0.01
    q_noise_std: float = 0.0
    meas_noise_std: float = 0.04

    #State estimation measurement noise sigma 
    sigma: float = 0.04

@dataclass
class PipelineOutputs:
    # Outputs from running the pipeline
    time: np.ndarray
    Z_clean: np.ndarray
    Z_attacked: np.ndarray
    attack_mask: np.ndarray
    converged: np.ndarray
    H: np.ndarray
    X_true: np.ndarray
    X_hat: np.ndarray
    residual_norms: np.ndarray
    metadata: Dict[str, Any]

def run_pipeline(
        net,
        config: PipelineConfig,
        scenario: ScenarioConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> PipelineOutputs:
    
    """
    IEEE-9 FDIA detection pipeline execution model.
    1. simulation of time-series measurements
    2. FDIA injection (standard, random, stealth)
    3. state estimation via WLS per timestep
    4. residual signal formation (residual norms)
    5. return outputs (ready for dataset building)
    """
    if config.network != "ieee9":
        raise ValueError("Only 'ieee9' network is supported currently.")
    
    if rng is None:
        rng = np.random.default_rng(config.seed)

    # 1. Simulation
    Z_clean, X_true, converged, H = run_time_series(
    net=net,
    T=config.T,
    p_noise_std=config.p_noise_std,
    q_noise_std=config.q_noise_std,
    meas_noise_std=config.meas_noise_std,
    rng=rng,
    seed=config.seed,
    )

    # 2. FDIA Injection
    Z_attacked, attack_mask = inject_fdi_time_series(
        Z=Z_clean,
        H=H,
        attack_type=scenario.attack_type,
        attacked_indices=scenario.attacked_indices,
        alpha=scenario.alpha,
        start=scenario.start,
        end=scenario.end,
        rng=rng,
        shift=scenario.shift,
        scale=scenario.scale,
        seed=config.seed,
    )
    # 3. State Estimation + residual signal (norms)
    residual_norms, X_hat = run_wls_time_series(
        Z=Z_attacked,
        H=H,
        sigma=config.sigma,
    )

    # 4. Outputs
    time = np.arange(Z_clean.shape[0], dtype=int)

    metadata: Dict[str, Any] = {
        "network": config.network,
        "seed": config.seed,
        "T": int(config.T),
        "p_noise_std": float(config.p_noise_std),
        "q_noise_std": float(config.q_noise_std),
        "meas_noise_std": float(config.meas_noise_std),
        "sigma": float(config.sigma),
        "attack_type": scenario.attack_type,
        "attack_start": int(scenario.start),
        "attack_end": int(scenario.end),
        "attack_shift": float(scenario.shift),
        "attack_scale": float(scenario.scale),
        "attack_alpha": float(scenario.alpha),
        "attacked_indices": None if scenario.attacked_indices is None else np.asarray(scenario.attacked_indices).tolist(),
        "shapes": {
            "Z_clean": list(Z_clean.shape),
            "Z_attacked": list(Z_attacked.shape),
            "X_true": list(X_true.shape),
            "X_hat": list(X_hat.shape),
            "H": list(H.shape),
            "attack_mask": list(attack_mask.shape),
            "converged": list(converged.shape),
            "residual_norms": list(residual_norms.shape),
        },
    }
    return PipelineOutputs(
        time=time,
        Z_clean=Z_clean,
        Z_attacked=Z_attacked,
        attack_mask=attack_mask,
        converged=converged,
        H=H,
        X_true=X_true,
        X_hat=X_hat,
        residual_norms=residual_norms,
        metadata=metadata,
    )