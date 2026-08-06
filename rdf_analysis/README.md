RDF Analysis of Nickel and Cobalt Ions in SPC/E Water under Electric Fields

Overview

This repository contains molecular dynamics simulations and analysis scripts used to calculate the radial distribution functions (RDFs) between nickel or cobalt ions and water oxygen atoms under different applied electric-field strengths.

The main quantities analyzed are:

- Ni–Ow RDF
- Co–Ow RDF
- First-shell coordination numbers (CNs) of both ions
- Changes in the hydration structure of the metal ions under different E-field strengths

Directory Structure

rdf_analysis/ 
├── md_analysis/ 
└── scripts/

md_analysis/

This directory contains the input files required to run molecular dynamics simulations 
of nickel and cobalt under different E-fields ranging from 0 to 3.0 nm/V

scripts/

This directory contains Python scripts used to calculate and analyze the RDF between the metal ions and water oxygen atoms, as well as the coordination numbers (CNs) of the first hydration shell.

Force Fields
- water: SPC/E
- nicke/cobalt: Babu, C. S.; Lim, C. *J. Phys. Chem. A* **2006**, *110*, 691–699.
- sulfate anion: Cannon, W. R.; Pettitt, B. M.; McCammon, J. A. *J. Phys. Chem.* **1994**, *98*, 6225–6230.



