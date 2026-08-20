This repository contains a Gaussian-process Bayesian optimization workflow for modeling and optimizing cobalt/nickel separation using pulsed electrodeposition methods.

##Model scope

The model uses physically informed pulse features, Gaussian-process uncertainty, a deposition surrogate, and a stacked deployed pipeline to recommend new experimental candidate parameters.

##Model workflow

1. Engineer pulsed electrodeposition features.
2. Train a controls-only Gaussian-process selectivity model.
3. Train a deposition surrogate from controllable experimental inputs.
4. Train a selectivity model using predicted deposition.
5. Compare baseline and stacked candidate rankings.
6. Run acquisition function ablations.
7. Validate model-recommended candidates experimentally.