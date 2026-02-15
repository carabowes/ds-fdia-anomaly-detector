import numpy as np
import pandapower as pp

class OPFController:
    """
    Simple redispatch controller that reacts to a specific bus angle
    (or a fallback norm) to guarantee observable physical impact.
    """

    def __init__(self, ramp_limits, attack_bus=None, gain=20.0):
        self.ramp_limits = ramp_limits
        self.attack_bus = attack_bus
        self.gain = float(gain)

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
            signal = float(np.linalg.norm(x_hat))

        # --- Differential redispatch (creates flow changes) ---
        gen_p = net.gen["p_mw"].to_numpy(dtype=float)

        gen_p_new = gen_p.copy()
        delta = self.gain * signal

        if len(gen_p_new) >= 2:
            gen_p_new[0] = gen_p[0] + delta
            gen_p_new[1] = gen_p[1] - delta
        else:
            gen_p_new[0] = gen_p[0] + delta

        return {"gen_p": gen_p_new}

# import pandapower as pp
# import numpy as np

# class OPFController:
#     def __init__(self, ramp_limits):
#         self.ramp_limits = ramp_limits

#     def compute_control(self, x_hat, net, t):
#         # x_hat = voltage angles (DC SE)
#         # simple heuristic: rebalance generation slightly
#         # based on average angle deviation

#         angle_bias = float(np.mean(x_hat))

#         gen_p_new = []
#         for i in range(len(net.gen)):
#             p_old = net.gen.at[i, "p_mw"]
#             p_new = p_old - 5.0 * angle_bias  # small feedback
#             gen_p_new.append(p_new)

#         return {"gen_p": np.array(gen_p_new)}

# def make_net_opf_ready(net):
#     net.gen["controllable"] = True
#     net.gen["min_p_mw"] = net.gen["p_mw"] - 50
#     net.gen["max_p_mw"] = net.gen["p_mw"] + 50

#     pp.create_poly_cost(
#         net,
#         element="gen",
#         element_index=net.gen.index,
#         cp1_eur_per_mw=1.0,
#     )