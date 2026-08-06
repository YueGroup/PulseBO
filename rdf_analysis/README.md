# RDF Analysis of Nickel and Cobalt Ions in SPC/E Water under Electric Fields

## Overview

This repository contains molecular dynamics simulation input files and analysis scripts used to investigate the hydration structure of Ni²⁺ and Co²⁺ ions in SPC/E water under applied electric fields. The analysis focuses on radial distribution functions (RDFs) between the metal ions and water oxygen atoms, as well as coordination numbers and hydration-shell properties.

The main quantities analyzed include:

- Ni–O<sub>w</sub> RDFs
- Co–O<sub>w</sub> RDFs
- First-shell coordination numbers (CNs) of nickel and cobalt
- Changes in the first hydration shell and overall hydration structure under different applied electric-field strengths

## Directory Structure

```text
rdf_analysis/
├── README.md
├── md_analysis/
│   ├── no-efield/
│   │   ├── cobalt/
│   │   └── nickel/
│   └── efields/
│       ├── cobalt/
│       └── nickel/
└── scripts/
```

### `md_analysis/`

This directory contains the input files required to perform molecular dynamics simulations of Ni²⁺ and Co²⁺ ions in aqueous solution under applied electric fields ranging from **0 to 3.0 V/nm**. It includes simulation setups for both zero-field and finite-field conditions.

### `scripts/`

This directory contains Python scripts used to:

- Calculate radial distribution functions (RDFs) between the metal ions and water oxygen atoms.
- Compute first-shell coordination numbers (CNs) from the RDFs.
- Analyze changes in the hydration structure under different applied electric fields.

## Force Fields

- **Water (SPC/E):** Berendsen, H. J. C.; Grigera, J. R.; Straatsma, T. P. *J. Phys. Chem.* **1987**, *91*, 6269–6271.
- ** Nicke/cobalt:**  Babu, C. S.; Lim, C. *J. Phys. Chem. A* **2006**, *110*, 691-699.
- **Sulfate anion :** Cannon, W. R.; Pettitt, B. M.; McCammon, J. A. *J. Phys. Chem.* **1994**, *98*, 6225-6230.



