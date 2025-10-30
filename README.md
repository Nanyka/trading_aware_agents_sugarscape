# Trading-Aware Agents in Sugarscape

**Author:** Binh Lai  
**Affiliation:** University of Vaasa, Finland  
**Correspondence:** laithanh@uwasa.fi  

This repository accompanies the paper  
**“Trading-Aware Agents in Sugarscape: A Deep Reinforcement Learning Approach to Adaptive Economic Behavior”**  
(submitted to *Computational Economics*, Springer).

---

## 🏗️ Project Overview

This project builds upon Epstein & Axtell’s *Sugarscape* model and extends it using modern **Deep Reinforcement Learning (DRL)** techniques implemented in **Unity ML-Agents**.

Agents are trained to:
- Gather and trade resources (sugar and spice),
- Maximize individual or survival-oriented rewards,
- Demonstrate emergent trading behaviors and welfare optimization.

Two independent reward schemes are provided:
1. **Cobb–Douglas Utility Reward** – Optimizes agents’ long-term utility following a smooth economic utility curve.  
2. **Kinked Survival Reward** – Emphasizes short-term survival with nonlinear welfare responses.

---

## 🧩 Project Structure

```
.
├── config/                              # ML-Agents configuration files
│   ├── cobb_douglas_reward.yaml
│   └── kinked_survival_reward.yaml
│
├── training_env/                        # Training environments
│   └── training_env_silicon.zip         # Unzip the file to use it
│
├── test_env_with_trained_model/         # Environment with pretrained model
│   └── test_env_with_trained_model_silicon.zip
│
├── test_env_with_import_model/          # Environment to test imported models
│   ├── test_env_with_import_model_silicon.zip
│   └── README.txt (instructions for reviewers)
│
├── onnx2sentis/                         # ONNX → Sentis converter
│   └── build/, src/, CMakeLists.txt
│
├── results/                             # Default output logs and checkpoints
│
├── TradingAware_Sugarscape_Replication/         # Analysis and visualization notebooks
│   ├── analyze_sugarscape_training_submission.ipynb
│   ├── data_visualizer.py
│   ├── simulation_manager.py
│   ├── images/
│   ├── env_for_generate_data/           # The simulation environment used for data generation
│   └── submission_data/                  
│
├── requirements.txt                     # Python dependencies
├── README.md                            # This document
└── LICENSE
```

---

## ⚙️ Environment Setup

### ✅ Create & activate a virtual environment
This guide provides a reproducible path to train a Unity ML-Agents project using Conda environments.  
It is suitable for reviewers on macOS (Apple Silicon or Intel) and Linux.

```bash
conda create -n training_env python=3.9.23 -y
conda activate training_env
pip install -r requirements.txt 
cd <project_local_directory>
```

---

## 🧠 Train Agents

### Cobb–Douglas Utility Reward
```bash
mlagents-learn ./config/cobb_douglas_reward.yaml    --env=./training_env/training_env_silicon.app   --run-id=cobb_douglas_run --no-graphic
```

### Key parameters (adjustable in YAML)
- **Population:** 500 agents  
- **Vision radius:** 10  
- **Max steps:** 5 × 10⁶  
- **Reward scheme:** {CobbDouglasUtility, KinkedSurvivalUtility}

Use descriptive run IDs for clarity, e.g. `cd_v1`, `ks_v1`.  
Optional flags:  
`--results-dir ./runs` (custom output path), `--force` (overwrite an existing run).

---

## 📈 Monitor Training with TensorBoard

To visualize learning curves and performance metrics in real time:

```bash
tensorboard --logdir results
```

Then open the displayed local URL (e.g., `http://localhost:6006`) in your browser.  
This allows you to compare training progress between the two reward schemes.

---

## 🧪 Test Trained and Imported Models

After training completes, two environments are provided to review or test models.

### ✅ 1. `test_env_with_trained_model`
This environment includes **pre-trained models**.  
Open the folder `test_env_with_trained_model` and run the executable  
(e.g., `test_env_with_trained_model_silicon`).  
Use this to observe the stable, fully trained behavior of agents under both reward schemes.

---

### ⚙️ 2. `test_env_with_import_model`
This environment allows reviewers to **test their own models** after completing the training steps.

Steps:
1. **Convert your trained model**  
   Use the included `onnx2sentis` tool to convert your exported `.onnx` model into `.sentis` format.
   This produces a file `policy.sentis`.

2. **Import and test**  
   Launch the environment `test_env_with_import_model_silicon` and press **“Import Model”**.  
   Select your `.sentis` file to assign it to the agents.

> ⚠️ **Note:**  
> Unity currently does **not** support importing `.onnx` models at runtime.  
> Therefore, this environment uses **Heuristic Mode** for Sentis inference.  
> In this mode, agents’ actions are predicted sequentially via Sentis, which introduces some delay in their responses.  
> Use this setup **only** to verify your model loads and behaves correctly, or to compare performance before and after training.  
> For smooth real-time behavior demonstrations, use the **`test_env_with_trained_model`** environment instead.

---

## 🧰 Reproducibility Summary

| Step | Description | Output |
|------|--------------|---------|
| 1 | Create Conda environment | Isolated Python environment |
| 2 | Train with ML-Agents | ONNX models & TensorBoard logs |
| 3 | Convert ONNX → Sentis | `.sentis` model for Unity runtime |
| 4 | Test pre-trained models | `test_env_with_trained_model` |
| 5 | Import custom model | `test_env_with_import_model` |

---

## 📊 Analysis and Visualization

The **`analysis/`** folder contains Jupyter notebooks and helper scripts for post-training evaluation and visualization:

- **`analyze_sugarscape_training_submission.ipynb`** — Main analysis notebook for visualizing agent performance, trading patterns, and welfare outcomes.  
- **`data_visualizer.py`** — Utility module for loading logs, plotting learning curves, and generating comparative charts across simulations.  
- **`simulation_manager.py`** — Controls batch simulations and manages experiment metadata.  
- **`images/`** — Contains plots and figures generated from experiments.
- **`submission_data/`** — Contains the simulation environment used for data generation (unzip prior to execution)
- **`submission_data/`** — Stores processed data used for journal submission figures.

---

## 🧩 Technical Notes

- **Engine:** Unity 6.1 + Sentis 2.1  
- **Backend:** GPUCompute (recommended)  
- **OS Support:** macOS (Apple Silicon)
- **Framework:** Unity ML-Agents 3.0.0  
- **Languages:** C#, Python  
