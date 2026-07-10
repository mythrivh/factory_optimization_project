import pulp
from src.model import build_supply_chain_model

regions = ["Interior", "Atlantic", "Gulf", "Pacific"]

capacity = {
    "Lot's O' Nuts": 5000,
    "Wicked Choccy's": 4000,
    "Sugar Shack": 4500,
    "Secret Factory": 3000,
    "The Other Factory": 3500
}

demand = {
    "Interior": 8820,
    "Atlantic": 11159,
    "Gulf": 6209,
    "Pacific": 12466
}

shipping_costs = {
    ("Lot's O' Nuts", "Interior"): 4,
    ("Lot's O' Nuts", "Atlantic"): 6,
    ("Wicked Choccy's", "Atlantic"): 5,
    ("Sugar Shack", "Interior"): 7,
    ("Sugar Shack", "Gulf"): 6,
    ("Secret Factory", "Pacific"): 8,
    ("The Other Factory", "Gulf"): 5,
    ("The Other Factory", "Pacific"): 7
}

model, allocations, unmet = build_supply_chain_model(capacity, demand, shipping_costs, regions)

print("Status:", pulp.LpStatus[model.status])
print("Objective value:", pulp.value(model.objective))
print("Allocations:", allocations)
print("Unmet demand:", unmet)
