from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandapower as pp
import json
import time
import pandapower.networks as pn
from dataclasses import asdict
from typing import Dict, Any, Sequence, Optional, Tuple, List
from collections import deque
from pathlib import Path

from src.pipeline.simulation import build_dc_measurement_model, simulate_measurements
from src.pipeline.state_estimation import wls_estimate
from src.pipeline.attack_schedule import generate_random_attack
from src.pipeline.attacks import stealth_FDIA, standard_FDIA, random_attack, make_bus_targeted_c

from src.pipeline.run_pipeline import PipelineConfig, ScenarioConfig

from src.control.opf_controller import OPFController
from src.control.apply_control import apply_control

from src.pipeline.attacks import raised_cosine_envelope




Episode = Dict[str, int]

class JSONLineWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = open(path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._f.write(json.dumps(record) + "\n")

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush()
        finally:
            self._f.close()

class EpisodeSchedule:
    def __init__(
        self,
        *,
        rng_seed: int,
        p_start: float,
        duration_min: int,
        duration_max: int,
        cooldown: int,
        no_attack_before: int,
        initial_horizon: int = 5000,
        extend_by: int = 5000,
    ):
        self._rng = np.random.default_rng(int(rng_seed))
        self._episodes: List[Dict[str, int]] = []

        self.p_start = float(p_start)
        self.duration_min = int(duration_min)
        self.duration_max = int(duration_max)
        self.cooldown = int(cooldown)
        self.no_attack_before = int(no_attack_before)

        self.horizon = 0
        self.extend_by = int(extend_by)
        
        # Internal pointer for incremental generation
        self._t_cursor = 0

        self._extend(initial_horizon)

    @property
    def episodes(self) -> List[Episode]:
        return list(self._episodes)

    def _extend(self, new_horizon: int) -> None:
        """
        Extend schedule to cover up to new_horizon (exclusive) without changing existing episodes.
        """
        if new_horizon <= self.horizon:
            return

        t = self._t_cursor

        # Ensure we don't start attacks before no_attack_before
        if t < self.no_attack_before:
            t = self.no_attack_before

        # Continue generating episodes forward in time
        while t < new_horizon:
            # Decide whether to start an attack at this timestep
            if self._rng.random() < self.p_start:
                dur = int(self._rng.integers(self.duration_min, self.duration_max + 1))
                start = t
                end = min(t + dur, new_horizon)

                # Record episode
                self._episodes.append({"start": int(start), "end": int(end)})

                # Jump forward past this episode + cooldown
                t = end + self.cooldown
            else:
                t += 1

        self.horizon = int(new_horizon)
        self._t_cursor = int(t)

    def ensure_coverage(self, t: int) -> None:
        # Extend schedule so that timestep t is within [0, horizon].
        while t >= self.horizon:
            self._extend(self.horizon + self.extend_by)
    
    def is_active(self, t: int) -> Tuple[bool, Optional[Dict[str, int]]]:
        # Returns whether an attack is active at timestep t.
        self.ensure_coverage(t)
        for ep in self._episodes:
            if ep["start"] <= t < ep["end"]:
                return True, {"start": ep["start"], "end": ep["end"]}
        return False, None
    
# Online innovation stream - need to add residual representation aswell for streaming consistency with offline pipeline
class InnovationStream:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self._z_hat_prev: Optional[np.ndarray] = None
        self._z_prev: Optional[np.ndarray] = None

    def step(self, z_t: np.ndarray) -> np.ndarray:
        z_t = np.asarray(z_t, dtype=float).reshape(-1)

        if self._z_hat_prev is None:
            self._z_hat_prev = z_t.copy()
            self._z_prev = z_t.copy()
            return np.zeros_like(z_t)

        z_hat_t = self.alpha * self._z_prev + (1.0 - self.alpha) * self._z_hat_prev
        e_t = z_t - z_hat_t

        self._z_hat_prev = z_hat_t
        self._z_prev = z_t.copy()
        return e_t

class OnlineWindowDetector:
    """
    Keeps a rolling window of per-timestep feature vectors and emits alarm/score.

    This is predict-only: training is offline.
    """
    def __init__(self, detector: Any, window_size: int, feature_dim: int, scaler: Any = None):
        if not hasattr(detector, "predict"):
            raise RuntimeError("Streaming requires detector.predict()")

        self.detector = detector
        self.scaler = scaler
        self.W = int(window_size)
        self.d = int(feature_dim)
        self.buf: deque[np.ndarray] = deque(maxlen=self.W)

    def update(self, feature_t: np.ndarray) -> Tuple[bool, float]:
        ft = np.asarray(feature_t, dtype=float).reshape(self.d)
        self.buf.append(ft)

        if len(self.buf) < self.W:
            return False, float("nan")

        X = np.concatenate(list(self.buf), axis=0).reshape(1, self.W * self.d)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        out = self.detector.predict(X)

        alarm = bool(out["alarms"][0])
        score_raw = out["scores"][0]
        score = float(score_raw) if score_raw is not None else float("nan")
        return alarm, score

    
def step_streaming(
    net,
    *,
    t: int,
    base_p: np.ndarray,
    base_q: Optional[np.ndarray],
    config: PipelineConfig,
    scenario: ScenarioConfig,
    rng_load: np.random.Generator,
    rng_meas: np.random.Generator,
    rng_attack: np.random.Generator,
    attack_active: bool,
    active_ep: Optional[Episode],
    attack_strength: float,
    c_direction: Optional[np.ndarray],
    attack_envelope: str,
) -> Dict[str, Any]:
    """
    One timestep:
      - perturb loads
      - run PF
      - build H,x_true
      - simulate clean measurement z_clean
      - apply attack -> z_attacked
      - run WLS on clean and attacked z
      - compute residual vectors + norms
    """

    # 1) Load perturbation (same idea as run_time_series.py)
    p_noise = rng_load.normal(0.0, config.p_noise_std, size=len(base_p))
    net.load.loc[:, "p_mw"] = base_p * (1.0 + p_noise)

    if base_q is not None and config.q_noise_std > 0.0:
        q_noise = rng_load.normal(0.0, config.q_noise_std, size=len(base_q))
        net.load.loc[:, "q_mvar"] = base_q * (1.0 + q_noise)

    # 2) Power flow
    # converged = True
    try:
        pp.runpp(net, algorithm="nr", init="dc", calculate_voltage_angles=True)
        converged = True
    except Exception:
        converged = False

    if not converged:
        return {"t": t, "converged": False}

    line_flows = net.res_line.p_from_mw.to_numpy().tolist()

    # 3) DC model
    H, x_true, _, _ = build_dc_measurement_model(net)

    # 4) Clean measurements
    z_clean = simulate_measurements(H, x_true, config.meas_noise_std, rng_meas)
    z_clean = np.asarray(z_clean, dtype=float).reshape(-1)
    
    # 5) Attack layer
    z_att = z_clean.copy()

    # # Determine attacked indices
    # if attack_active and scenario.attack_type == "stealth" and scenario.attacked_indices is None:
    #     raise ValueError("Stealth FDIA requires explicit attacked_indices")

    attacked_indices = (
        np.arange(len(z_clean))
        if scenario.attacked_indices is None
        else np.asarray(scenario.attacked_indices, dtype=int)
    )
    attacked_indices = np.atleast_1d(attacked_indices).astype(int)

    # # --- compute envelope ONCE (correct for random schedules) ---
    attack_env = 0.0
    max_delta = 0.0
    # if scenario.attack_type == "stealth" and active_ep is not None:
    #     attack_env = float(raised_cosine_envelope(t, active_ep["start"], active_ep["end"]))


    # Apply attack
    if attack_active:
        if scenario.attack_type == "standard":
            z_att = standard_FDIA(z_att, attacked_indices, scenario.shift)

        elif scenario.attack_type == "random":
            z_att = random_attack(z_att, attacked_indices, rng_attack, scenario.scale)

        elif scenario.attack_type == "stealth":

            if active_ep is None:
                raise RuntimeError("attack_active=True but active_ep is None")

            # Generate c_direction once per episode
            if c_direction is None:
                c_direction = make_bus_targeted_c(
                    n_state=H.shape[1],
                    attack_buses=scenario.attack_buses,
                    slack_bus=0,
                    rng=rng_attack,
                )
            assert H.shape[0] == len(z_clean)
            assert H.shape[1] == len(c_direction)


            # attack_env = float(
            #     raised_cosine_envelope(t, active_ep["start"], active_ep["end"])
            # )

            # percent = float(attack_strength) * attack_env
            if attack_envelope == "raised_cosine":
                attack_env = float(
                    raised_cosine_envelope(t, active_ep["start"], active_ep["end"])
                )
            elif attack_envelope == "none":
                attack_env = 1.0 if (active_ep["start"] <= t < active_ep["end"]) else 0.0
            else:
                raise ValueError("Invalid attack_envelope")

            percent = float(attack_strength) * attack_env

            z_att, a = stealth_FDIA(
                H=H,
                z_clean=z_clean,
                percent=percent,
                c_direction=c_direction,
            )

            max_delta = float(np.max(np.abs(a)))
        debug_attack = None
        if scenario.attack_type == "stealth":
            debug_attack = {
                "attack_env": float(attack_env),
                "percent": float(percent),
                "max_abs_a": float(np.max(np.abs(a))) if attack_active else 0.0,
}


        # elif scenario.attack_type == "stealth":

        #     if active_ep is None:
        #         raise RuntimeError("attack_active=True but active_ep is None")

        #     # Generate direction ONCE per episode
        #     if c_direction is None:
        #         n = H.shape[1]
        #         c_direction = rng_attack.standard_normal(n)
        #         c_direction = c_direction / np.linalg.norm(c_direction)

        #     attack_env = float(
        #         raised_cosine_envelope(t, active_ep["start"], active_ep["end"])
        #     )

        #     percent = float(attack_strength)

        #     a = stealth_FDIA(
        #         H=H,
        #         z_clean=z_clean,
        #         attacked_indices=attacked_indices,
        #         percent=percent,
        #         c_direction=c_direction,
        #     )

        #     z_att = z_att + attack_env * a
        #     # z_att = z_att + a

        # delta_vec = z_att - z_clean
        # max_delta = float(np.max(np.abs(delta_vec)))

    # 6) State estimation on clean measurements 
    x_hat_clean, _ = wls_estimate(H, z_clean, config.sigma)
    r_clean = z_clean - H @ x_hat_clean
    resid_norm_clean = float(np.linalg.norm(r_clean))

    # 7) SE (attacked measurements)
    x_hat_att, _ = wls_estimate(H, z_att, config.sigma)
    r_att = z_att - H @ x_hat_att
    resid_norm_att = float(np.linalg.norm(r_att))

    if scenario.attack_type == "stealth" and attack_active and (t == scenario.start + 5):
        xdiff = x_hat_att - x_hat_clean
        print(f"[DEBUG t={t}] attack_env={debug_attack['attack_env']:.4f} "
            f"percent={debug_attack['percent']:.6f} "
            f"max|a|={debug_attack['max_abs_a']:.6e} "
            f"max|x_diff|={np.max(np.abs(xdiff)):.6e}")
    # if t % 50 == 0:
        # print("max |x_true|:", np.max(np.abs(x_true)))

    return {
        "t": t,
        "converged": True,
        "H": H,
        "x_true": x_true,
        "z_clean": z_clean,
        "z_attacked": z_att,
        "x_hat_clean": x_hat_clean,
        "x_hat_attacked": x_hat_att,
        # Residual vectors (needed for streaming residual rep consistency)
        "r_clean": r_clean,
        "r_attacked": r_att,
        # residual norms (still useful for plots/logging)
        "residual_norm_clean": resid_norm_clean,
        "residual_norm_attacked": resid_norm_att,

        "attack_max_delta": float(max_delta),
        "attack_active": bool(attack_active),
        "attack_envelope": float(attack_env),
        "line_flows": line_flows,

        "c_direction": c_direction

    }

def run_streaming_pipeline(
    net,
    *,
    config: PipelineConfig,
    scenario: ScenarioConfig,
    out_root: Path,
    detector: Optional[Any] = None,
    scaler: Optional[Any] = None,
    window_size: int = 5,
    representation: str = "innovations", # "innovations" or "residuals"
    innovation_alpha: float = 0.7,
    attack_schedule_mode: str = "fixed",  # "fixed" or "random"
    p_start: float = 0.03,
    duration_min: int = 5,
    duration_max: int = 40,
    cooldown: int = 10,
    no_attack_before: int = 200,
    stop_after_steps: Optional[int] = None,
    controller: Optional[OPFController] = None,
    control_on_alarm: bool = False,
    log_features: bool = False,
    attack_strength: float = 1.0,
    attack_envelope: str = "raised_cosine",
    enable_mitigation: bool = False,
    mitigation_mode: str = "freeze",   # only "freeze" supported here
    # calibration_steps: Optional[int] = None # if None -> auto picks safe window
    enable_recovery: bool = False, # whether to run recovery controller after attack ends

) -> Path:
    """
    Produces 3 JSONL files aligned by timestep:
      - clean.jsonl                 (clean measurements + x_true)
      - attacked_measurements.jsonl (attacked measurement stream)
      - attacked_estimates.jsonl    (x_hat on attacked + residual norm )

    Closed-loop:
      - detector runs online using innovations (alpha) and windowing (window_size)
      - controller optionally uses attacked x_hat to compute control u_t
      - apply_control(net, u_t) modifies net before next timestep
      - if controller is provided, apply_control(net, u_t) modifies net for next timestep
    """

    if config.network != "ieee9":
        raise ValueError("Only 'ieee9' network is supported currently.")
    # if representation != "innovations":
    #     raise ValueError("Streaming currently supports ONLY innovations (aligned with offline pipeline).")
    if representation not in ["innovations", "residuals", "state_derivative", "state"]:
        raise ValueError("representation must be 'innovations' or 'residuals' or 'state_derivative' or 'state'")
    
    if mitigation_mode != "freeze":
        raise ValueError("Only mitigation_mode='freeze' is supported")
    
    # RNG discipline:
    # keep load + measurement noise reproducible and independent from attack randomness
    rng_load = np.random.default_rng(int(config.seed))
    rng_meas = np.random.default_rng(int(config.seed) + 1)
    rng_attack = np.random.default_rng(int(config.seed) + 3)

    # Attack scheduling
    schedule: Optional[EpisodeSchedule] = None
    if attack_schedule_mode == "random":
        schedule = EpisodeSchedule(
            rng_seed=config.seed + 2,
            p_start=p_start,
            duration_min=duration_min,
            duration_max=duration_max,
            cooldown=cooldown,
            no_attack_before=no_attack_before,
        )

    elif attack_schedule_mode != "fixed":
        raise ValueError("attack_schedule_mode must be 'fixed' or 'random'")

    # Output dir layout: runs_live/ieee9/...
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / "ieee9" / scenario.attack_type / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    clean_w = JSONLineWriter(run_dir / "clean.jsonl")
    meas_w = JSONLineWriter(run_dir / "attacked_measurements.jsonl")
    est_w = JSONLineWriter(run_dir / "attacked_estimates.jsonl")

    feature_writer = JSONLineWriter(run_dir / "features.jsonl") if log_features else None

    # Baseline loads
    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy() if config.q_noise_std > 0 else None

    innov_stream = InnovationStream(alpha=innovation_alpha)
    predictor: Optional[OnlineWindowDetector] = None
    x_hat_prev: Optional[np.ndarray] = None

    # mitigation state
    z_trusted: Optional[np.ndarray] = None
    trusted_frozen: bool = False

    # ---------------- Recovery State (Alarm-only, Realistic) ----------------
    gen_p_baseline = (
        net.gen["p_mw"].to_numpy().copy()
        if len(net.gen) > 0
        else None
    )
    
    # recovery_enabled = controller is not None
    # recovery_enabled = False
    recovery_enabled = enable_recovery and (controller is not None)
    recovery_active = False
    ever_controlled = False
    emergency_latched = False
    clean_streak = 0

    cooldown_steps = 10          # alarm-free steps before recovery starts
    max_recovery_step = 5.0      # MW per timestep
    recovery_tol = 1e-2          # stop threshold

    # if controller is not None:
    #     pass
    
    
    active_ep = None
    current_attack_direction: Optional[np.ndarray] = None
    current_episode_id: Optional[Tuple[int, int]] = None
    episode_control_used = False
    calib_q = 0.99
    calib_scores: List[float] = []
    calibrated = False
    calibrated_threshold: Optional[float] = None

    calibration_end_t = int(scenario.start) if attack_schedule_mode == "fixed" else int(no_attack_before)

    t = 0
    prev_attack_active = False
    try:
        while True:
            if attack_schedule_mode == "fixed":
                attack_active = scenario.start <= t < scenario.end
                active_ep = {"start": scenario.start, "end": scenario.end} if attack_active else None
                safe_clean = t < scenario.start
            else:
                assert schedule is not None
                attack_active, active_ep = schedule.is_active(t)
                safe_clean = t < no_attack_before
            
            # Reset clean streak when a new attack begins
            if attack_active and not prev_attack_active:
                clean_streak = 0
                episode_control_used = False


            prev_attack_active = attack_active
             
            if scenario.attack_type == "stealth":
                if attack_active:
                    ep_id = (active_ep["start"], active_ep["end"])
                    if current_episode_id != ep_id:
                        current_episode_id = ep_id
                        current_attack_direction = None
                else:
                    current_episode_id = None
                    current_attack_direction = None
            
            if isinstance(current_attack_direction, np.ndarray):
                c_dir = current_attack_direction
            else:
                c_dir = None

            step = step_streaming(
                net,
                t=t,
                base_p=base_p,
                base_q=base_q,
                config=config,
                scenario=scenario,
                rng_load=rng_load,
                rng_meas=rng_meas,
                rng_attack=rng_attack,
                attack_active=attack_active,
                active_ep=active_ep,
                attack_strength=attack_strength,
                attack_envelope=attack_envelope,
                c_direction=c_dir,
                
            )

            if not step["converged"]:
                t += 1
                continue

            if enable_mitigation and not trusted_frozen:
                if safe_clean:
                    # Guaranteed-clean window → keep updating trusted
                    z_trusted = step["z_attacked"].copy()
                else:
                    # First timestep outside safe window → freeze forever
                    if z_trusted is not None:
                        trusted_frozen = True
            if scenario.attack_type == "stealth":
                current_attack_direction = step.get("c_direction", None)

            #  clean baseline stream (now includes clean SE outputs)
            clean_w.write({
                "t": t,
                "z": step["z_clean"].tolist(),
                "x_true": step["x_true"].tolist(),
                "x_hat": step["x_hat_clean"].tolist(),
                "residual_norm": float(step["residual_norm_clean"]),
                "gen_p_mw": (
                    net.gen["p_mw"].to_numpy().tolist()
                    if len(net.gen) > 0
                    else None
                ),
                "line_flows": step.get("line_flows", None),
            })
            gen_p_pre = (
                net.gen["p_mw"].to_numpy().tolist()
                if len(net.gen) > 0
                else None
            )
            line_flows_pre = step.get("line_flows", None)

            # attacked measurements stream
            meas_w.write({
                "t": t,
                "attack_active": bool(attack_active),
                "active_episode": active_ep,
                "z_attacked": step["z_attacked"].tolist(),
            })

            alarm = False
            score: Optional[float] = None
            z_feature = step["z_attacked"]

            if detector is not None or log_features:


                if representation == "innovations":
                    feature_t = innov_stream.step(z_feature)

                elif representation == "residuals":                
                    feature_t = np.array([step["residual_norm_attacked"]])
                    # feature_t = step["r_attacked"]
                    # feature_t = np.abs(step["r_attacked"]) 

                elif representation == "state":

                    feature_t = step["x_hat_attacked"].reshape(-1)

                   
                else:
                    raise RuntimeError("Invalid representation")

                if log_features and feature_writer is not None:
                    feature_writer.write({"t": t, "features": feature_t.tolist()})

                if detector is not None:
                    if predictor is None:
                        predictor = OnlineWindowDetector(
                            detector=detector,
                            window_size=window_size,
                            feature_dim=int(feature_t.shape[0]),
                            scaler=scaler,
                        )

                    model_alarm, score_val = predictor.update(feature_t)
                    score = None if np.isnan(score_val) else float(score_val)

                    # CALIBRATION LOGIC
                    # 1) While in guaranteed-clean region, collect scores (once we have valid score).
                    if (not calibrated) and safe_clean and (score is not None):
                        calib_scores.append(score)

                    # 2) When we reach the end of guaranteed-clean region, freeze calibrated threshold once.
                    #    We do this the FIRST timestep that is NOT safe_clean.
                    if (not calibrated) and (not safe_clean):
                        if len(calib_scores) >= 20:
                            calibrated_threshold = float(np.quantile(np.asarray(calib_scores), calib_q))
                            
                        else:
                            # Fallback if not enough calibration scores (shouldn't happen unless tiny runs)
                            calibrated_threshold = None
                        calibrated = True

                    # 3) Decide alarm:
                    #    - before calibration is frozen: force alarm False (prevents huge early FPR)
                    #    - after calibration: compare score to calibrated threshold
                    if not calibrated:
                        alarm = False
                    else:
                        if calibrated_threshold is None or score is None:
                            # fallback: use detector’s own alarm if calibration failed
                            alarm = bool(model_alarm)
                        else:
                            alarm = bool(score >= calibrated_threshold)

            # Mitigation Application
            z_used = step["z_attacked"].copy()
            z_used_source = "attacked"
            mitigation_applied = False

            if enable_mitigation and alarm and trusted_frozen:
                z_used = z_trusted.copy()
                z_used_source = "trusted_frozen"
                mitigation_applied = True
            
            # Estimate using the measurement that actually drives control/metrics
            x_hat_used, _ = wls_estimate(step["H"], z_used, config.sigma)
            r_used = z_used - step["H"] @ x_hat_used
            resid_norm_used = float(np.linalg.norm(r_used))

            
        # # Closed-loop controller hook
        #     u_t: Optional[Dict[str, Any]] = None
        #     applied_control = False
        #     control_reason: Optional[str] = None
        #     line_flows_post = None
        #     gen_p_post = None
        #     if controller is not None:
        #         # Decide whether to apply control this step
        #         if control_on_alarm:
        #             should_control = alarm
        #             control_reason = "alarm_triggered" if alarm else "no_alarm"
        #         else:
        #             should_control = True
        #             control_reason = "always_control"
        #         # line_flows_post = None
        #         # gen_p_post = None

        #         if should_control:
        #             # x_hat_used, _ = wls_estimate(step["H"], z_used, config.sigma)
        #             u_t = controller.compute_control(
        #                 x_hat=x_hat_used,
        #                 # x_hat=step["x_hat_attacked"],
        #                 net=net,
        #                 t=t,
        #             )

        #             apply_control(
        #                 net,
        #                 u_t,
        #                 ramp_limits=getattr(controller, "ramp_limits", None),
        #             )
        #             applied_control = True

        #             # RE-RUN PHYSICS AFTER CONTROL
        #             try:
        #                 pp.runpp(net, algorithm="nr", init="dc", calculate_voltage_angles=True)

        #                 line_flows_post = net.res_line.p_from_mw.to_numpy().tolist()
        #                 gen_p_post = (
        #                     net.gen["p_mw"].to_numpy().tolist()
        #                     if len(net.gen) > 0
        #                     else None
        #                 )

        #             except Exception as e:
        #                 # keep post values None, but DO NOT crash the run
        #                 applied_control = True  # attempted
        #                 line_flows_post = None
        #                 gen_p_post = None
        #                 # optional: log error string if you want (not required)
        #                 # control_reason = f"{control_reason}|pf_fail"
                        
        #                 # If PF fails, keep post values None
        #                 # line_flows_post = None
        #                 # gen_p_post = None
        #     # post_pf_converged = True

        # ---------------- Closed-loop controller + recovery ----------------
            u_t: Optional[Dict[str, Any]] = None
            applied_control = False
            control_reason: Optional[str] = None
            line_flows_post = None
            gen_p_post = None

            if controller is not None:

                # ---------------- EMERGENCY CONTROL ----------------
                # if control_on_alarm and alarm:

                #     clean_streak = 0
                #     recovery_active = False

                #     u_t = controller.compute_control(
                #         x_hat=x_hat_used,
                #         net=net,
                #         t=t,
                #     )

                #     apply_control(
                #         net,
                #         u_t,
                #         ramp_limits=getattr(controller, "ramp_limits", None),
                #     )

                #     applied_control = True
                #     ever_controlled = True
                #     control_reason = "alarm_triggered"
                
                #Update for state representation
                if control_on_alarm and alarm and not emergency_latched and not episode_control_used:

                    clean_streak = 0
                    recovery_active = False

                    u_t = controller.compute_control(
                        x_hat=x_hat_used,
                        net=net,
                        t=t,
                    )

                    apply_control(
                        net,
                        u_t,
                        ramp_limits=getattr(controller, "ramp_limits", None),
                    )

                    emergency_latched = True
                    ever_controlled = True
                    episode_control_used = True
                    control_reason = "alarm_triggered_once"

                    try:
                        pp.runpp(net, algorithm="nr", init="dc", calculate_voltage_angles=True)
                        line_flows_post = net.res_line.p_from_mw.to_numpy().tolist()
                        gen_p_post = net.gen["p_mw"].to_numpy().tolist()
                    except Exception:
                        line_flows_post = None
                        gen_p_post = None

                # ---------------- RECOVERY ----------------
                else:

                    if recovery_enabled:

                        #  Track clean streak from detector
                        if not alarm:
                            clean_streak += 1
                        else:
                            clean_streak = 0
                            recovery_active = False

                        #  Only evaluate recovery if we have baseline
                        if gen_p_baseline is not None:

                            gen_p_current = net.gen["p_mw"].to_numpy(dtype=float)

                            # Check if system is still off nominal
                            off_nominal_tol = 0.05
                            off_nominal = np.max(np.abs(gen_p_current - gen_p_baseline)) > off_nominal_tol

                            # Residual must be small (system looks physically consistent)
                            small_threshold = 0.05  # adjust based on clean residual scale
                            system_clean = resid_norm_used < small_threshold

                            #  Trigger recovery
                            # if (
                            #     ever_controlled
                            #     and clean_streak >= cooldown_steps
                            #     and system_clean
                            #     and off_nominal
                            # ):
                            if (
                                ever_controlled
                                and clean_streak >= cooldown_steps
                                and off_nominal
                            ):
                                recovery_active = True

                        #  Perform gradual ramp back
                        if recovery_active and gen_p_baseline is not None:

                            gen_p_current = net.gen["p_mw"].to_numpy(dtype=float)

                            delta = gen_p_baseline - gen_p_current

                            # Limit ramp step
                            step_delta = np.clip(delta, -max_recovery_step, max_recovery_step)

                            gen_p_next = gen_p_current + step_delta

                            u_rec = {"gen_p": gen_p_next}

                            apply_control(
                                net,
                                u_rec,
                                ramp_limits=getattr(controller, "ramp_limits", None),
                            )

                            applied_control = True
                            control_reason = "recovery_ramp"

                            try:
                                pp.runpp(net, algorithm="nr", init="dc", calculate_voltage_angles=True)
                                line_flows_post = net.res_line.p_from_mw.to_numpy().tolist()
                                gen_p_post = net.gen["p_mw"].to_numpy().tolist()
                            except Exception:
                                line_flows_post = None
                                gen_p_post = None

                            #  Stop recovery once baseline reached
                            if np.max(np.abs(gen_p_next - gen_p_baseline)) < recovery_tol:
                                recovery_active = False
                                emergency_latched = False


            est_w.write(
                {
                    "t": t,
                    # Estimates
                    "x_hat_attacked": step["x_hat_attacked"].tolist(),
                    "x_hat_used": x_hat_used.tolist(),
                    "residual_norm_attacked": float(step["residual_norm_attacked"]),
                    "residual_norm_used": float(resid_norm_used),
                    # Detection
                    "alarm": bool(alarm),
                    "score": score,
                    # Mitigation
                    "mitigation_enabled": bool(enable_mitigation),
                    "mitigation_applied": bool(mitigation_applied),
                    "z_used_source": z_used_source,
                    "trusted_frozen": bool(trusted_frozen),
                    # "calibration_end": int(calibration_end),
                    "trusted_initialized": bool(z_trusted is not None),
                    # Control
                    "control_applied": bool(applied_control),
                    "control_reason": control_reason,
                    "recovery_active": bool(recovery_active),
                    "clean_streak": int(clean_streak),
                    "u_t": (
                        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in u_t.items()}
                        if u_t is not None
                        else None
                    ),
                    "gen_p_pre": gen_p_pre,
                    "line_flows_pre": line_flows_pre,
                    "gen_p_post": gen_p_post,
                    "line_flows_post": line_flows_post,
                    # Attack Diagnostics
                    "attack_active": bool(attack_active),
                    "attack_max_delta": step.get("attack_max_delta", 0.0),
                    "attack_envelope": step.get("attack_envelope", None),
                    
                }
            )

            t += 1
            if stop_after_steps is not None and t >= stop_after_steps:
                break

    except KeyboardInterrupt:
        post_pf_converged = False
        print("[LIVE STOPPED] Keyboard interrupt received")
    finally:
        clean_w.close()
        meas_w.close()
        est_w.close()
        if feature_writer is not None:
            feature_writer.close()
        (run_dir / "stop.json").write_text(json.dumps({"stopped_at_t": t}, indent=2))

    return run_dir