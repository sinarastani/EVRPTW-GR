
---

## Problem Description

All three models solve the same EVRPTW-GR:

- Electric vehicles with limited battery capacity
- Time windows for customers
- Charging stations with a partial recharging strategy
- Load-dependent energy consumption
- Road gradients and regenerative braking
- Objective: minimize total energy consumption

---

## Model Descriptions

### MILP1 — Approximate Energy Model

- Uses an approximate energy consumption formulation
- Fastest model
- May fail to find feasible solutions in challenging instances
- Intended for preliminary analysis only

---

### MILP2 — Linearized Accurate Model (Recommended)

- Fully linearized formulation of the EVRPTW-GR
- Accurately captures:
  - Load-dependent energy consumption
  - Road gradients
  - Regenerative braking
- Robust and reliable
- Produces optimal solutions
- Best trade-off between accuracy and runtime

---

### MILP3 — Exact State-of-Charge Tracking Model

- Tracks battery State-of-Charge (SoC) explicitly
- Computes objective value using remaining battery and recharging amounts
- Slowest model
- Produces the same optimal solutions as MILP2
- Included for validation and methodological completeness

---

## Data

The `Data/` directory contains extended EVRPTW benchmark instances:

- Based on classical EVRPTW instances
- Augmented with altitude information
- Three terrain types:
  - Level
  - Nearly Level
  - Very Gentle

The data is **part of this research contribution**.

@article{EVRPTWGR,
  title   = {Uphill Struggles, Downhill Gains: How Road Gradients and Load Dynamics Influence Electric Vehicle Routing Decisions},
  author  = {Rastani, Sina and Keskin, Merve and Yüksel, Tuğçe and Çatay, Bülent},
  journal = {To appear},
  year    = {XXXX}
}
---

## How to Run

### Requirements
- Python ≥ 3.9
- Gurobi Optimizer (academic or commercial license)
- NumPy, Pandas

### Execution
Edit the desired instance inside the corresponding MILP file and run:


python MILP2.py

### Results

Full computational results are provided in the "Routes Information.xlsx."


### License and Usage Rights
## Academic Use

Allowed only with proper citation of the above paper.

## Commercial Use

Not permitted without explicit written permission from the authors.

## Data Usage

The provided datasets are part of this research and must be cited if reused.

### Contact

For questions, collaboration, or commercial licensing:

Sina Rastani
University of Sheffield
s.rastani@sheffield.ac.uk

