# PTA_FourierLikelihood

In this repository we stored all the main scripts used for the project described in the paper:

["Moving towards a pulsar timing array likelihood in Fourier space"](https://arxiv.org/abs/2412.11894).

where we introduce a PTA likelihood in Fourier space. This new formulation allows to divide a GWB search in a two-steps analysis, where all the signals not covaraint with a GWB are evaluated separately for each pulsar and then marginalized over when analysing the whole array. This likelihood formulation can also be used for a GWB search on a Gamma-ray PTA dataset.

If you make use of any of these scripts, please cite:
```
@misc{valtolina2024regularizingpulsartimingarray,
      title={Regularizing the Pulsar Timing Array likelihood: A path towards Fourier Space}, 
      author={Serena Valtolina and Rutger van Haasteren},
      year={2024},
      eprint={2412.11894},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2412.11894}, 
}
```

## Repository content details

- **FourierCoeff_reconstruct.ipynb** : tutorial about how to obtain the best estimate for the Fourier coefficients describing red noise and DM variations using the python package *la_forge*.
- **SPNA_example.ipynb** : example of single pulsar noise analysis for the EPTA pulsar J1738+0333 with the Fourier likelihood (Fig. 1 of our paper).
- **GWBsearch_FourierLikelihood_tutorial.ipynb** : tutorial about how to use the Fourier likelihood for a GWB search on a PTA dataset.
- **GWB_FourierLikelihood.py** : code for a GWB search on a PTA dataset with the Fourier likelihood (Fig. 2 of our paper).
- **PCA_analysis.ipynb** : tutorial related to Appendix B of our paper: possible strategy to marginlaize over DM variation hyperparameters when doing a PTA analysis with a PCA approach.
- **sere_enterprise.py** : slightly modified version of some enterprise extension function. 

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
