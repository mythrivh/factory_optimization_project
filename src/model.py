# =========================
# model.py
# =========================
import pulp

def build_supply_chain_model(capacity, demand, shipping_costs, regions):
    """
    Build and solve the supply chain optimization model.

    Parameters:
    - capacity: dict {factory: capacity units}
    - demand: dict {region: demand units}
    - shipping_costs: dict {(factory, region): cost per unit}
    - regions: list of region names

    Returns:
    - model: solved PuLP model
    - allocations: list of dicts {Factory, Region, Quantity}
    - unmet: dict {region: unmet demand}
    """

    # Define model
    model = pulp.LpProblem("Factory_Reallocation", pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("ship",
                              [(f, r) for f in capacity for r in regions],
                              lowBound=0,
                              cat="Integer")

    slack = pulp.LpVariable.dicts("unmet", regions, lowBound=0, cat="Integer")

    # Penalty for unmet demand
    penalty = 10 * max(shipping_costs.values())

    # Objective function
    model += (
        pulp.lpSum(
            shipping_costs[(f, r)] * x[(f, r)]
            for f in capacity for r in regions if (f, r) in shipping_costs
        )
        + pulp.lpSum(penalty * slack[r] for r in regions)
    )

    # Demand constraints
    for r in regions:
        model += (
            pulp.lpSum(x[(f, r)] for f in capacity if (f, r) in shipping_costs)
            + slack[r] >= demand[r]
        )

    # Capacity constraints
    for f in capacity:
        model += (
            pulp.lpSum(x[(f, r)] for r in regions if (f, r) in shipping_costs)
            <= capacity[f]
        )

    # Solve
    model.solve()

    # Collect allocations
    allocations = []
    for (f, r) in x:
        val = x[(f, r)].value()
        if val is not None and val > 0:
            allocations.append({"Factory": f, "Region": r, "Quantity": int(val)})

    # Collect unmet demand
    unmet = {r: int(slack[r].value()) if slack[r].value() else 0 for r in regions}

    return model, allocations, unmet
