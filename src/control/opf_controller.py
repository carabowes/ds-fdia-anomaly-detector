import numpy as np
import pandapower as pp

class OPFController:
    """
    Simple redispatch controller that reacts to a specific bus angle
    (or a fallback norm) to guarantee observable physical impact.
    """

    def __init__(self, ramp_limits, attack_bus=None, gain=5.0, signal_clip=0.5):
        self.ramp_limits = ramp_limits
        self.attack_bus = attack_bus
        self.gain = float(gain)
        self.signal_clip = float(signal_clip)

    def compute_control(self, x_hat, net, t):
        """
        x_hat: DC state estimate (angles with slack removed)
        net: pandapower network
        """

        # --- Choose a non-cancelling control signal ---
        if self.attack_bus is not None and self.attack_bus > 0:
            # DC SE removes slack → index = bus_id - 1
            idx = self.attack_bus - 1
            signal = float(x_hat[idx])
        else:
            # Fallback: never cancels like mean()
            # signal = float(np.linalg.norm(x_hat))
            signal = float(np.mean(np.abs(x_hat)))

        # --- Differential redispatch (creates flow changes) ---
        gen_p = net.gen["p_mw"].to_numpy(dtype=float)

        # Clip signal to avoid insane deltas if SE spikes
        signal = float(np.clip(signal, -self.signal_clip, self.signal_clip))

        gen_p = net.gen["p_mw"].to_numpy(dtype=float)
        gen_p_new = gen_p.copy()

        delta = self.gain * signal

        if len(gen_p_new) >= 2:
            gen_p_new[0] = gen_p[0] + delta

            # gen_p_new[1] = gen_p[1] - delta
        # else:
        #     gen_p_new[0] = gen_p[0] + delta

        return {"gen_p": gen_p_new}