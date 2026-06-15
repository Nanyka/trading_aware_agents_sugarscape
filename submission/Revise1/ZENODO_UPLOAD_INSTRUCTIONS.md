# Zenodo Update Instructions — Seed List Files

**DOI:** 10.5281/zenodo.17466369  
**Purpose:** Add the fixed evaluation seed list referenced in the revised manuscript (Revision 1).

## Files to Upload

Add these three files to the existing Zenodo record:

| File | Description |
|------|-------------|
| `evaluation_seeds.csv` | Seed list in CSV format (replicate, seed columns) |
| `evaluation_seeds.json` | Seed list in JSON format with full metadata |
| `generate_seeds.py` | Python script that regenerates the list from scratch |

## How the Seeds Were Generated

```python
import numpy as np
rng = numpy.random.default_rng(42)
seeds = rng.integers(low=0, high=2**31 - 1, size=50).tolist()
```

Master seed: **42**  
Number of seeds: **50** (one per evaluation replicate per condition)  
Library: `numpy` (any version ≥ 1.17 supporting `default_rng`)

## How Seeds Were Used

Each of the 50 seeds was passed as the environment random seed for one evaluation episode. The same 50 seeds were used across all four experimental conditions:

1. Rule-based agents — deterministic maps
2. DRL agents — deterministic maps  
3. Rule-based agents — stochastic maps
4. DRL agents — stochastic maps

This ensures that all conditions faced identical initial resource distributions and agent parameter draws within each replicate, making inter-condition comparisons valid.

## Steps to Update the Zenodo Record

1. Go to https://zenodo.org/record/17466369
2. Click **"New version"** (do not edit the existing published version directly)
3. Upload the three files above
4. In the version description / "What's new", write:
   > Added fixed evaluation seed list (evaluation_seeds.csv, evaluation_seeds.json, generate_seeds.py) as referenced in the revised manuscript submitted to Computational Economics (Revision 1).
5. Publish the new version. The DOI 10.5281/zenodo.17466369 will automatically resolve to the latest version.

> **Note:** The manuscript cites the Zenodo DOI, not a version-specific DOI, so no change to the .tex file is needed after uploading.
