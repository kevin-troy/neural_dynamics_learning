# Neural Assisted Control
Using neural networks to efficiently compute optimal control policies.

A project to implement data-driven control methods. An extension of the work performed in "[Optimal Energy Shaping via Neural Approximators](https://arxiv.org/abs/2101.05537)" by Massaroli et. al. 

The focus of this work is to extend the results of the paper from a single DOF pendulum to a multi-DOF system, namely a double pendulum.

- The baseline representation recreating the work of Massaroli et. al. can be found in the file "test_energy_shaping_pendulum.ipynb".

Status:
- [x] Baseline implementation with the help of [Torchdyn](https://github.com/DiffEqML/torchdyn) + tutorials recreating results of the paper.
- [x] Extraction of trajectories and animation of single-DOF pendulum
- [x] Cuda compatibility
- [ ] Extension to multi-DOF system(s)
- [ ] Error analysis and network structure refinement
- [ ] Comparison to vanilla linear Model-Predictive Control

<p align="center">
  <img src="https://github.com/kevin-troy/neural_optimal_control/blob/main/sandbox/swingup.gif" width="400" height="300" />
</p>
