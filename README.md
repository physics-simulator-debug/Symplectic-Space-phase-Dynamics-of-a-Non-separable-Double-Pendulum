# Symplectic Space-phase Dynamics of a Non-separable Double Pendulum

[![DOI](https://zenodo.org)](https://doi.org)
[![License: MIT](https://shields.io)](https://opensource.org)

This repository contains the source code for the numerical simulation and chaos analysis of a planar double pendulum system using a custom fourth-order symplectic integrator.

## Abstract
Traditional numerical integration techniques (e.g., RK45) fail to conserve the total mechanical energy over long-term chaotic simulations of conservative Hamiltonian systems. This project introduces a geometric numerical simulation platform using an implicit **Störmer-Verlet method** upgraded with **Yoshida’s fourth-order composition constants**. Using this energy-preserving platform, the **Benettin renormalization method** is applied to calculate the Maximal Lyapunov Exponent (MLE). The results provide a rigorous numerical proof of chaos ($ \lambda_{max} \approx 0.041 $) without artificial numerical damping.

## Core Features
* **Symplectic Integration:** Implementation of a customized implicit Störmer-Verlet algorithm to handle the non-separable Hamiltonian of the double pendulum.
* **Yoshida's Composition:** Elevates the local numerical truncation error from second-order $O(h^2)$ to fourth-order $O(h^4)$ precision.
* **Chaos Quantification:** Complete Benettin renormalization framework to compute the running Maximal Lyapunov Exponent (MLE).
* **Energy Stability:** Bounded energy error strictly confined within a high-frequency envelope ($\approx \pm10^{-4}$), eliminating long-term numerical drift.

## Mathematical Formulation

### 1. Symplectic Step Updates
The non-separable Hamiltonian forces an implicit three-step structure to advance the system states $(q, p)$ over a time step $h$:

$$\text{Step 1: } p_{n+\frac{1}{2}} = p_n - \frac{h}{2} \nabla_q H(q_n, p_{n+\frac{1}{2}})$$

$$\text{Step 2: } q_{n+1} = q_n + \frac{h}{2} \left[ \nabla_p H(q_n, p_{n+\frac{1}{2}}) + \nabla_p H(q_{n+1}, p_{n+\frac{1}{2}}) \right]$$

$$\text{Step 3: } p_{n+1} = p_{n+\frac{1}{2}} - \frac{h}{2} \nabla_q H(q_{n+1}, p_{n+\frac{1}{2}})$$

### 2. Benettin Renormalization
To calculate the MLE without phase-space saturation, the perturbed trajectory is rescaled back to the initial separation distance $d_0$ at regular intervals $t$:

$$\lambda = \frac{1}{Nt} \sum_{i=1}^{N} \ln\left(\frac{d_i}{d_0}\right)$$

## Installation & Setup
Clone this repository to your local machine:
```bash
git clone https://github.com
cd Symplectic-Space-phase-Dynamics-of-a-Non-separable-Double-Pendulum
```

### Dependencies
Make sure you have the following requirements installed:
* Python 3.x
* NumPy
* Matplotlib

Install the packages via pip:
```bash
pip install numpy matplotlib
```

## Usage
To run the main simulation platform and output both the total mechanical energy comparison plot and the MLE convergence curve, execute:
```bash
python main.py
```

## Results
* **Energy Conservation:** Bounded energy oscillations with no systemic directional drift over extended horizons, outperforming explicit classical solvers.
* **Chaotic Limit:** Asymptotic convergence of the MLE to a constant positive value of $\lambda \approx 0.041$, establishing absolute verification of deterministic chaos.

## Citation
If you use this code or paper in your research, please cite it as:
```bibtex
@article{gaafar2026symplectic,
  title={Symplectic Space-phase Dynamics of a Non-separable Double Pendulum},
  author={Gaafar, Yassin Gaafar Mohamed},
  journal={Department of Physics, Faculty of Science, Ain Shams University},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.20979592}
}
```

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
