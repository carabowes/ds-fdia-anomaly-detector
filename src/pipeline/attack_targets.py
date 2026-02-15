import numpy as np

IEEE9_ATTACK_BUSES = {
    "generator_bus": 1,
    "load_heavy_bus": 5,
    "generator_neighbour": 2,
    "electrically_central": 4,
}

def get_attack_buses():
    """
    Returns list of buses an informed attacker is willing to target.
    """
    return list(IEEE9_ATTACK_BUSES.values())
    
def select_multiple_attack_buses(net, roles=None):
    """
    roles: list like ["generator_bus", "load_heavy_bus"]
    """
    buses = choose_attack_buses_ieee9(net)

    if roles is None:
        roles = ["generator_bus", "load_heavy_bus"]

    selected = []
    for r in roles:
        if r not in buses:
            raise ValueError(f"Unknown role: {r}")
        selected.append(buses[r])

    return sorted(set(selected))

def choose_attack_buses_ieee9(net):
    # 1) Generator bus: pick the slack/ext_grid bus if present, else first gen bus
    if len(net.ext_grid) > 0:
        gen_bus = int(net.ext_grid.bus.iloc[0])
        gen_bus_role = "slack/ext_grid"
    else:
        gen_bus = int(net.gen.bus.iloc[0])
        gen_bus_role = "gen"

    # 2) Load-heavy bus: max p_mw load
    if len(net.load) == 0:
        raise ValueError("No loads found in net.load; cannot pick load-heavy bus.")
    idx = int(np.argmax(net.load.p_mw.to_numpy()))
    load_heavy_bus = int(net.load.bus.iloc[idx])

    # 3) Neighbour of generator: pick a bus connected by a line to gen_bus
    # (Use line endpoints; pick the first neighbour deterministically.)
    neighs = set()
    for _, row in net.line.iterrows():
        fb = int(row.from_bus)
        tb = int(row.to_bus)
        if fb == gen_bus:
            neighs.add(tb)
        elif tb == gen_bus:
            neighs.add(fb)
    if not neighs:
        raise ValueError(f"No line neighbours found for generator bus {gen_bus}.")
    gen_neighbour_bus = int(sorted(neighs)[0])

    # 4) Electrically central bus: simple topology closeness proxy (degree as fallback)
    # If you want better: use networkx + weights. This keeps it minimal/no new deps.
    deg = {int(b): 0 for b in net.bus.index.to_list()}
    for _, row in net.line.iterrows():
        deg[int(row.from_bus)] += 1
        deg[int(row.to_bus)] += 1

    # Avoid duplicates: don't pick one of the already selected buses
    chosen = {gen_bus, load_heavy_bus, gen_neighbour_bus}
    central_candidates = [(d, b) for b, d in deg.items() if b not in chosen]
    if not central_candidates:
        central_bus = int(gen_bus)  # fallback (shouldn't happen in IEEE-9)
    else:
        central_bus = int(sorted(central_candidates, reverse=True)[0][1])

    return {
        "generator_bus": gen_bus,
        "generator_bus_role": gen_bus_role,
        "load_heavy_bus": load_heavy_bus,
        "gen_neighbour_bus": gen_neighbour_bus,
        "central_bus": central_bus,
    }
