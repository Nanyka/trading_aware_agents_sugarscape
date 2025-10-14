# Trading-Aware Agents in Sugarscape

**Author:** Binh Lai  
**Affiliation:** University of Vaasa, Finland  
**Correspondence:** binh.lai@uwasa.fi  

This repository accompanies the paper  
**“Trading-Aware Agents in Sugarscape: A Deep Reinforcement Learning Approach to Adaptive Economic Behavior”**  
(submitted to *Computational Economics*, Springer).

---

## 🧭 Overview
This project extends the classical **Sugarscape** model by integrating **Deep Reinforcement Learning (DRL)** to jointly optimize movement and trading decisions.  
Agents learn adaptive strategies via **Proximal Policy Optimization (PPO)**, closing the traditional *“move–then–trade”* gap and producing emergent equilibria consistent with economic theory.

### Key Features
- **Environment:** Unity-based Sugarscape world with dual renewable resources (sugar & spice).  
- **Learning Algorithm:** Parameter-sharing PPO with centralized training and decentralized execution.  
- **Behavioral Regimes:**  
  1. *Cobb–Douglas Utility Scheme* — welfare-maximizing consumption behavior.  
  2. *Kinked Survival Utility Scheme* — lexicographic survival-first behavior.  
- **Metrics:** Carrying capacity, market-price stability, welfare efficiency, and inequality (Gini & Pareto indices).  
- **Policy Experiments:** Optional transaction-tax analysis (Appendix A of the paper).

---

## 📁 Repository Structure
```
Sugarscape-DRL/
│
├── README.md                 ← this file
├── LICENSE
│
├── training_env/              ← Unity training environment
│   ├── training_env_64bit
│   └── training_env_silicon
│
├── trained_models/
│   ├── cobb_douglas_reward.onnx      ← trained model for cobbcobb_douglas_reward
│   └── kinked_survival_reward.onnx   ← trained model for kinked_survival_reward
│
├── config/
│   ├── cobb_douglas_reward.yaml  ← configuration for cobbcobb_douglas_reward
│   └── kinked_survival_reward.yaml ← configuration for kinked_survival_reward
│
└── requirements.txt
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Nanyka/trading_aware_agents_sugarscape.git
cd trading_aware_agents_sugarscape
```

### 2️⃣ Install dependencies
```bash
pip install -r training/requirements.txt
```
*Required packages:* `mlagents`, `torch`, `numpy`, `matplotlib`, `pandas`, `seaborn`, `notebook`.

### 3️⃣ Create & activate a virtual environment

This guide provides a **reproducible** path to train a Unity ML‑Agents project using **Conda** environments. It is suitable for reviewers who need a clean setup on macOS (Apple Silicon or Intel) or Linux.

```bash
conda create -n py3923 python=3.9.23 -y
conda activate py3923
pip install -r requirements.txt 
cd <project_local_directory>
```

---

## 🚀 Usage

### 🧠 Train agents

**Cobb-Douglas Utility Reward**:
```bash
mlagents-learn ./config/cobb_douglas_reward.yaml  --env=./training_env/training_env_silicon.app --run-id=<RUN_ID> --no-graphic
```

**Kinked Survival Reward**:
```bash
mlagents-learn ./config/kinked_survival_reward.yaml  --env=./training_env/training_env_silicon.app --run-id=<RUN_ID> --no-graphic
```

Key parameters (also adjustable in `config.yaml`):
- Population = 500 agents  
- Vision = 10  
- Max steps = 5 × 10⁶  
- Reward scheme = {CobbDouglasUtility | KinkedSurvival}

Use a descriptive value for <RUN_ID> to distinguish and save checkpoints/models for each experiment (e.g., cd_utility_v1, kinked_survival_ablation).
Optional flags you may add: --results-dir ./runs (custom output path), --force (overwrite an existing run).

### 📊 Run analysis
After training, open the notebook:
```bash
jupyter notebook analysis/analysis_notebook.ipynb
```
The notebook reproduces:
- Carrying-capacity curves  
- Price-stability plots  
- Welfare and inequality metrics  
- Pareto-tail dynamics and spatial patterns  

---

## 📈 Main Results (summary)
| Metric | DRL vs. Rule-based | Description |
|---------|--------------------|--------------|
| **Carrying capacity** | ↑ 28–32 % | DRL agents sustain larger populations. |
| **Price volatility** | ↓ ~50 % | Faster convergence to equilibrium. |
| **Aggregate welfare** | ↑ 7 % | More efficient resource utilization. |
| **Inequality (Gini)** | ↓ 0.05–0.08 | Fairer long-run wealth distribution. |

---

## 🧩 Reproducibility Notes
- Deterministic and stochastic resource landscapes are both supported.  
- Each reported result averaged over 50 replications with fixed random seeds.  
- Training reproducible via `train.py` using the provided config.  
- Hardware used: NVIDIA RTX 3090 (24 GB), Intel i9-13900K, 64 GB RAM.  

---

## 🧠 Citation
```bibtex
@article{Lai2025SugarscapeDRL,
  author  = {Binh Lai},
  title   = {Trading-Aware Agents in Sugarscape: A Deep Reinforcement Learning Approach to Adaptive Economic Behavior},
  journal = {Computational Economics},
  year    = {2025},
  note    = {Submitted manuscript},
}
```

---

## 📜 License
This project is released under the **MIT License** — see [LICENSE](LICENSE).

---

## 📂 Data Availability Statement
The Unity simulation environment, training scripts, and analysis notebooks used in the paper are publicly available at  
👉 **https://github.com/BinhLai/trading_aware_agents_sugarscape**  
(commit `v1.0_submission`).

---

## 🤝 Acknowledgements
This work was supported by the **University of Vaasa** and the **DigiConsumers Research Network**.  
The author thanks **Prof. Panu Kalmi** and the *Computational Economics* editorial team for constructive feedback on reproducibility and open-science practices.
