# Co/Ni DFT Dataset

This directory contains density functional theory (DFT) calculations for nickel(II) and cobalt(II) ion complexation with boric acid and borate species. The dataset was generated to investigate the thermodynamics of borate coordination to hydrated metal ions and to support the analysis presented in the associated manuscript.

## Computational Methods

All DFT calculations were performed using **ORCA 5.0.4**.

- Density functional: M06-2X with D3 dispersion correction
- Geometry optimization: def2-SVP basis set
- Single-point energy calculations: def2-TZVP basis set
- Implicit solvation: CPCM (water)
- Vibrational frequency calculations were performed to obtain thermochemical corrections.

The thermochemical quantities include:

- Electronic energies
- Zero-point energy (ZPE)
- Enthalpy
- Entropy
- Gibbs free energy

To account for solvent-constrained librational motion, the translational and rotational entropy contributions were reduced by 50%.

## Directory Structure

- `borate/` — Reference calculations for borate species
- `boric_molecule/` — Reference calculations for boric acid species
- `water/` — Reference calculations for water molecules
- `cobalt/` — Cobalt(II) complexes coordinated with boric acid and borate ligands
- `cobalt_water/` — Hydrated cobalt(II) reference complexes
- `nickel/` — Nickel(II) complexes coordinated with boric acid and borate ligands
- `nickel_water/` — Hydrated nickel(II) reference complexes

Each calculation directory contains:

- ORCA input files (`.inp`)
- ORCA output files (`.out`)
- Optimized geometries (`.xyz`)
- Wavefunction files (`.gbw`)
- Electronic energies (energy folder)
- Vibrational frequency calculations
- Thermochemical corrections (ZPE, enthalpy and entropy)

## Purpose

This dataset accompanies the study of borate coordination to hydrated nickel(II) and cobalt(II) ions and provides the computational data used to analyze the relative thermodynamic stability of boric acid- and borate-coordinated complexes.
