[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# CacheTest

This archive is distributed in association with the INFORMS Journal on Computing under the MIT License.

The software and data in this repository are a snapshot of the software and data used in the research reported in the paper Single- and Dual-Policy Reinforcement Learning for Dynamic Bike Rebalancing by Jiaqi Liang, Defeng Liu, Sanjay Dominik Jena, Andrea Lodi, and Thibaut Vidal. The repository contains the implementation and data used for the computational experiments reported in the paper.


## Cite

To cite the contents of this repository, please cite both the paper and this repository, using their respective DOIs.

Paper DOI: https://doi.org/10.1287/ijoc.2025.1756

Code repository DOI: https://doi.org/10.1287/ijoc.2025.1756.cd

Below is the BibTeX for citing this snapshot of the repository.


```
@misc{BikeRebalancingRL,
  author = {Jiaqi Liang, Defeng Liu, Sanjay Dominik Jena, Andrea Lodi, and Thibaut Vidal},
  publisher = {INFORMS Journal on Computing},
  title = {Single- and Dual-Policy Reinforcement Learning for Dynamic Bike Rebalancing},
  year = {2026},
  doi = {10.1287/ijoc.2025.1756.cd},
  url = {https://github.com/INFORMSJoC/2025.1756},
  note = {Available for download at https://github.com/INFORMSJoC/2025.1756}
}
```

## Description

This repository contains the implementation and data for computational experiments on dynamic bike-sharing rebalancing. The research investigates reinforcement learning methods for making vehicle routing and bike inventory decisions in a dynamic, stochastic bike-sharing environment.

Two approaches are studied: single-policy reinforcement learning (SPRL), which jointly learns routing and inventory decisions, and dual-policy reinforcement learning (DPRL), which uses separate policies for routing and inventory decisions. The methods are evaluated using an event-driven simulation framework and compared with benchmark approaches.


## Requirement

The code requires Python and the packages used by the simulation environment and reinforcement learning implementation.

Please refer to the dependency files included in this repository for the exact package versions and installation instructions.

## Data

The data directory contains the datasets used in the computational experiments, organized into the G1 and G2 subdirectories.

These data are used to define the problem instances for the computational experiments. Please consult the paper and the accompanying code to determine the role of each dataset and its corresponding experimental configuration.
