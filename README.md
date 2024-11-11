# PTA_FourierLikelihood

In this repository we stored all the main scripts used for the project described in the paper:

["Moving towards a pulsar timing array likelihood in Fourier space"](??).

where we introduce a PTA likelihood in Fourier space. This new formulation allows to divide a GWB search in a two-steps analysis, where all the signals not covaraint with a GWB are evaluated separately for each pulsar and then marginalized over when analysing the whole array. This likelihood formulation can also be used for a GWB search on a Gamma-ray PTA dataset.

If you make use of any of these scripts/data, please cite:
```
ADD
```

## Repository content details

- **FourierCoeff_reconstruct.ipynb**
- **SPNA_example.ipynb**
- **GWBsearch_FourierLikelihood_tutorial.ipynb**
- **GWB_FourierLikelihood.py**
- **PCA_analysis.ipynb**
- **sere_enterprise.py**

## Before running

Those scripts use the EPTA branch of ENTERPRISE (Enhanced Numerical Toolbox Enabling a Robust PulsaR Inference SuitE), a pulsar timing analysis code aimed at noise analysis, gravitational-wave searches, and timing model analysis. 
```
git clone https://gitlab.in2p3.fr/epta/enterprise_extensions/
git clone https://gitlab.in2p3.fr/epta/enterprise/
```
The python package *la_forge* is also needed [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4152550.svg)](https://doi.org/10.5281/zenodo.4152550) .
```
pip install la-forge
```
The dataset used for this work is the data published as EPTA DR2new [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10276364.svg)](https://doi.org/10.5281/zenodo.10276364).
