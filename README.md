# AE 695 Vehicle Lateral Dynamics Project

This project implements the nonlinear dynamic bicycle model with a stiffness-matched simplified Pacejka tire model. It includes equilibrium and linearization analysis, open-loop validation, canonical realizations, controllability and observability, state-feedback control, full-order and reduced-order observers, nonlinear closed-loop simulation, noise studies and a reproducible PDF report.

## Requirements

- Python 3.10 or newer
- NumPy, SciPy, Matplotlib, pandas, SymPy and ReportLab

```powershell
py -m pip install -r requirements.txt
```

## Run the complete project

```powershell
py main.py
```

The master script creates result files under `results/`, including figures, CSV tables and simulation data, and writes `report/project_report.pdf`.

## Run tests

```powershell
py -m pytest -q
```

## Structure

The `models`, `analysis`, `controllers`, `observers`, `simulations`, `visualization`, `report` and `tests` directories separate the implementation. `config.py` contains the central vehicle parameters.

## Main assumptions

Constant longitudinal speed, planar motion, no load transfer or roll/pitch, and equivalent axle forces are assumed. The report discusses the model limitations and the operating-region dependence of nonlinear results.
