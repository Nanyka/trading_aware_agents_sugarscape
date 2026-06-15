"""
generate_seeds.py
-----------------
Reproduces the fixed random seed list used for all policy evaluation replicates
in Lai (2025), "Trading-Aware Movement in Sugarscape: A Deep Reinforcement
Learning Approach", Computational Economics.

Usage:
    python generate_seeds.py

Outputs:
    Prints all 50 seeds to stdout. These match evaluation_seeds.csv and
    evaluation_seeds.json in this replication package.
"""

import numpy as np

MASTER_SEED = 42      # master seed fixed at 42
N_REPLICATES = 50     # 50 replicates per experimental condition

rng = np.random.default_rng(MASTER_SEED)
seeds = rng.integers(low=0, high=2**31 - 1, size=N_REPLICATES).tolist()

print(f"Fixed evaluation seeds (master seed = {MASTER_SEED}):")
print(f"{'Replicate':>10}  {'Seed':>12}")
print("-" * 25)
for i, s in enumerate(seeds, start=1):
    print(f"{i:>10}  {s:>12}")
