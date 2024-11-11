import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import scipy.stats as sstats

import os, sys, glob, json, types
import dill as pickle
from itertools import product

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise.pulsar import Pulsar
from enterprise.signals import selections
from enterprise.signals.selections import no_selection, Selection

from enterprise.signals import parameter, white_signals, utils, gp_signals, signal_base
from enterprise.signals.parameter import function
import enterprise.constants as const
from tqdm import tqdm

import la_forge.gp as lfgp
import scipy.linalg as sl
from numpy import linalg as LA

import arviz as az
import xarray as xr
import corner

from enterprise_extensions import blocks, model_orfs
from enterprise_extensions import chromatic as e_chrom

from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp
from enterprise.signals import (gp_signals, parameter, selections, utils,
                                white_signals)

from enterprise.signals import deterministic_signals
from enterprise_extensions import chromatic as chrom
from enterprise_extensions import dropout as drop
from enterprise_extensions import gp_kernels as gpk


########## powerlaw flat tail
from enterprise.signals.parameter import function
@function
def powerlaw_flat_tail(f, log10_A=-16, gamma=5, log10_kappa=-7, components=2):
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    pl = (10**log10_A) ** 2 / 12.0 / np.pi**2 * const.fyr ** (gamma - 3) * f ** (-gamma) * np.repeat(df, components)
    flat = 10 ** (2*log10_kappa)
    return np.maximum(pl, flat)


########### chromatic noise
def chromatic_noise_block(gp_kernel='nondiag', psd='powerlaw',
                          nondiag_kernel='periodic',
                          prior='log-uniform', name='chrom',
                          include_quadratic=False, idx=4,
                          dt=15, df=200, Tspan=None,
                          logf=False, fmin=None, fmax=None,
                          components=30, tnfreq=False,
                          select=None, modes=None, 
                          gamma_prior='uniform',gamma_val=None, logA_val=None,logk_val=None,
                          gammamin=0, gammamax=7,
                          delta_val=None, coefficients=False,
                          tndm=False, logmin=None, logmax=None,
                          dropout=False, dropbin=False, 
                          dropbin_min=10, k_threshold=0.5):
    """
    Returns GP chromatic noise model :

        1. Chromatic modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']

    :param gp_kernel:
        Whether to use a diagonal kernel for the GP. ['diag','nondiag']
    :param nondiag_kernel:
        Which nondiagonal kernel to use for the GP.
        ['periodic','sq_exp','periodic_rfband','sq_exp_rfband']
    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']
    :param prior:
        What type of prior to use for amplitudes. ['log-uniform','uniform']
    :param dt:
        time-scale for linear interpolation basis (days)
    :param df:
        frequency-scale for linear interpolation basis (MHz)
    :param idx:
        Index of radio frequency dependence (i.e. DM is 2). Any float will work.
    :param include_quadratic:
        Whether to include a quadratic fit.
    :param name: Name of signal
    :param Tspan:
        Tspan from which to calculate frequencies for PSD-based GPs.
    :param components:
        Number of frequencies to use in 'diag' GPs.
    :param coefficients:
        Whether to keep coefficients of the GP.
    :param dropout: Use a dropout analysis for chromatic noise models.
        Currently only supports power law option.
    :param dropbin: Use a dropout analysis for the number of frequency bins.
        Currently only supports power law option.
    :param dropbin_min: Set the minimal number of freq. bins for the dropbin.
    :param k_threshold: Threshold for dropout analysis.
    """
    if tnfreq and Tspan is not None:
        components = blocks.get_tncoeff(Tspan, components)
    if idx is None:
        idx = parameter.Uniform(0, 7)
    if gp_kernel=='diag':
        if tndm:
            chm_basis = gpb.createfourierdesignmatrix_dm_tn(nmodes=components,
                                                            Tspan=Tspan, logf=logf,
                                                            fmin=fmin, fmax=fmax,
                                                            idx=idx, modes=modes)
        else:
            chm_basis = gpb.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                Tspan=Tspan, logf=logf,
                                                                fmin=fmin, fmax=fmax,
                                                                idx=idx, modes=modes)
        if psd in ['powerlaw', 'turnover', 'broken_powerlaw', 'flat_powerlaw']:
            if logA_val is not None:
                log10_A = parameter.Constant(logA_val)
            elif logmin is not None and logmax is not None:
                if prior == 'uniform':
                    log10_A = parameter.LinearExp(logmin, logmax)
                elif prior == 'log-uniform':
                    log10_A = parameter.Uniform(logmin, logmax)
                elif prior == 'gaussian':
                    log10_A = parameter.Normal(logmin, logmax)
            else:
                if prior == 'uniform':
                    log10_A = parameter.LinearExp(-20, -10)
                elif prior == 'log-uniform':
                    log10_A = parameter.Uniform(-20, -10)
            
            if gamma_val is not None:
                gamma = parameter.Constant(gamma_val)
            else: 
                if gamma_prior == 'uniform':
                    gamma = parameter.Uniform(gammamin, gammamax)
                elif gamma_prior == 'gaussian':
                    gamma = parameter.Normal(gammamin, gammamax)

            # PSD
            if psd == 'powerlaw':
                if any([dropout, dropbin]):
                    if dropout:
                        k_drop = parameter.Uniform(0, 1)
                    else:
                        k_drop = 1
                    if dropbin:
                        k_dropbin = parameter.Uniform(dropbin_min, components+1)
                    else:
                        k_dropbin = None
                    chm_prior = drop.dropout_powerlaw(log10_A=log10_A, gamma=gamma, 
                                                      k_drop=k_drop, k_dropbin=k_dropbin,
                                                      k_threshold=k_threshold)
                else:
                    chm_prior = utils.powerlaw(log10_A=log10_A, gamma=gamma)
            elif psd == 'broken_powerlaw':
                kappa = parameter.Uniform(0.01, 0.5)
                log10_fb = parameter.Uniform(-10, -6)

                if delta_val is not None:
                    delta = parameter.Constant(delta_val)
                else:
                    delta = parameter.Uniform(0, 7)
                chm_prior = gpp.broken_powerlaw(log10_A=log10_A, gamma=gamma,
                                               delta=delta,
                                               log10_fb=log10_fb, kappa=kappa)
            elif psd == 'turnover':
                kappa = parameter.Uniform(0, 7)
                lf0 = parameter.Uniform(-9, -7)
                chm_prior = utils.turnover(log10_A=log10_A, gamma=gamma,
                                           lf0=lf0, kappa=kappa)
            elif psd == 'flat_powerlaw':
                if logk_val is not None:
                    log10_B = parameter.Constant(logk_val)
                else:
                    log10_B = parameter.Uniform(-10, -4)
                chm_prior = gpp.flat_powerlaw(log10_A=log10_A, gamma=gamma,
                                              log10_B=log10_B)

        if psd == 'spectrum':
            if logmin is not None and logmax is not None:
                if prior == 'uniform':
                    log10_rho = parameter.LinearExp(logmin, logmax,
                                                    size=components)
                elif prior == 'log-uniform':
                    log10_rho = parameter.Uniform(logmin, logmax,
                                                  size=components)
            else:
                if prior == 'uniform':
                    log10_rho = parameter.LinearExp(-10, -4, size=components)
                elif prior == 'log-uniform':
                    log10_rho = parameter.Uniform(-10, -4, size=components)
                else:
                    log10_rho = parameter.Uniform(-9, -4, size=components)

            chm_prior = gpp.free_spectrum(log10_rho=log10_rho)

    elif gp_kernel == 'nondiag':
        if nondiag_kernel == 'periodic':
            # Periodic GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            chm_basis = gpk.linear_interp_basis_chromatic(dt=dt*const.day)
            chm_prior = gpk.periodic_kernel(log10_sigma=log10_sigma,
                                            log10_ell=log10_ell,
                                            log10_gam_p=log10_gam_p,
                                            log10_p=log10_p)

        elif nondiag_kernel == 'periodic_rfband':
            # Periodic GP kernel for DM with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            log10_alpha_wgt = parameter.Uniform(-4, 1)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            chm_basis = gpk.get_tf_quantization_matrix(df=df, dt=dt*const.day,
                                                       dm=True, dm_idx=idx)
            chm_prior = gpk.tf_kernel(log10_sigma=log10_sigma,
                                      log10_ell=log10_ell,
                                      log10_gam_p=log10_gam_p,
                                      log10_p=log10_p,
                                      log10_alpha_wgt=log10_alpha_wgt,
                                      log10_ell2=log10_ell2)

        elif nondiag_kernel == 'sq_exp':
            # squared-exponential kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)

            chm_basis = gpk.linear_interp_basis_chromatic(dt=dt*const.day, idx=idx)
            chm_prior = gpk.se_dm_kernel(log10_sigma=log10_sigma,
                                         log10_ell=log10_ell)
        elif nondiag_kernel == 'sq_exp_rfband':
            # Sq-Exp GP kernel for Chrom with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            log10_alpha_wgt = parameter.Uniform(-4, 1)

            chm_basis = gpk.get_tf_quantization_matrix(df=df, dt=dt*const.day,
                                                       dm=True, dm_idx=idx)
            chm_prior = gpk.sf_kernel(log10_sigma=log10_sigma,
                                      log10_ell=log10_ell,
                                      log10_alpha_wgt=log10_alpha_wgt,
                                      log10_ell2=log10_ell2)

    if select is None:
        cgp = gp_signals.BasisGP(chm_prior, chm_basis, name=name+'_gp',
                                 coefficients=coefficients)
    else:
        cgp = gp_signals.BasisGP(chm_prior, chm_basis, name=name+'_gp',
                                 coefficients=coefficients,
                                 selection=select)

    if include_quadratic:
        # quadratic piece
        basis_quad = chrom.chromatic_quad_basis(idx=idx)
        prior_quad = chrom.chromatic_quad_prior()
        cquad = gp_signals.BasisGP(prior_quad, basis_quad, name=name+'_quad')
        cgp += cquad

    return cgp



############ DM exponential dip
def dm_exponential_dip(tmin, tmax, idx=2, sign='negative', name='dmexp', t0_dmexp_val=None, log10_Amp_dmexp_val=None, log10_tau_dmexp_val=None):
    """
    Returns chromatic exponential dip (i.e. TOA advance):

    :param tmin, tmax:
        search window for exponential dip time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sign:
        set sign of dip: 'positive', 'negative', or 'vary'
    :param name: Name of signal

    :return dmexp:
        chromatic exponential dip waveform.
    """
    if t0_dmexp_val is not None:
        t0_dmexp = parameter.Constant(t0_dmexp_val)
    else:
        t0_dmexp = parameter.Uniform(tmin, tmax)
    if log10_Amp_dmexp_val is not None:
        log10_Amp_dmexp = parameter.Constant(log10_Amp_dmexp_val)
    else:
        log10_Amp_dmexp = parameter.Uniform(-10, -2)
    if log10_tau_dmexp_val is not None:
        log10_tau_dmexp = parameter.Constant(log10_tau_dmexp_val)
    else:    
        log10_tau_dmexp = parameter.Uniform(0, 2.5)
    if sign == 'vary':
        sign_param = parameter.Uniform(-1.0, 1.0)
    elif sign == 'positive':
        sign_param = 1.0
    else:
        sign_param = -1.0
    wf = e_chrom.chrom_exp_decay(log10_Amp=log10_Amp_dmexp,
                         t0=t0_dmexp, log10_tau=log10_tau_dmexp,
                         sign_param=sign_param, idx=idx)
    dmexp = deterministic_signals.Deterministic(wf, name=name)

    return dmexp


######## common red noise block
def common_red_noise_block(psd='powerlaw', prior='log-uniform',
                           Tspan=None, psrTspan=True, components=30,
                           tnfreq=False, combine=True,
                           log10_A_val=None, gamma_val=None,
                           gamma_prior='uniform', gammamin=0, gammamax=7,
                           delta_val=None, logmin=None, logmax=None,
                           orf=None, orf_bins=None,
                           orf_ifreq=0, leg_lmax=5,
                           name='gw', coefficients=False, select=None,
                           logf=False, fmin=None, fmax=None,
                           modes=None, pshift=False, pseed=None,
                           dropout=False, dropout_psr='all', 
                           dropout_common=False, dropbin=False,
                           dropbin_psr='all', dropbin_common=False,
                           dropbin_min=10, k_threshold=0.5,
                           idx=None, tndm=False,
                           flagname='group', flagval=None):
    """
    Returns common red noise model:

        1. Red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum', 'broken_powerlaw']
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for individual pulsar.
    :param psrTspan: 
        Option to use pulsar time span. 
        Used only if sub-group of ToAs is chosen
    :param log10_A_val:
        Value of log10_A parameter for fixed amplitude analyses.
    :param gamma_val:
        Value of spectral index for power-law and turnover
        models. By default spectral index is varied of range [0,7]
    :param delta_val:
        Value of spectral index for high frequencies in broken power-law
        and turnover models. By default spectral index is varied in range [0,7].\
    :param logmin:
        Specify the lower bound of the prior on the amplitude for all psd but 'spectrum'.
        If psd=='spectrum', then this specifies the lower prior on log10_rho_gw
    :param logmax:
        Specify the lower bound of the prior on the amplitude for all psd but 'spectrum'.
        If psd=='spectrum', then this specifies the lower prior on log10_rho_gw
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].
    :param orf_ifreq:
        Frequency bin at which to start the Hellings & Downs function with
        numbering beginning at 0. Currently only works with freq_hd orf.
    :param leg_lmax:
        Maximum multipole of a Legendre polynomial series representation
        of the overlap reduction function [default=5]
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param name: Name of common red process
    :param dropout: Use a dropout analysis for common red noise models.
        Currently only supports power law option.
    :param dropbin: Use a dropout analysis for the number of frequency bins.
        Currently only supports power law option.
    :param dropbin_min: Set the minimal number of freq. bins for the dropbin.
    :param dropout_psr: Which pulsar to use a dropout switch on. The value 'all'
        will use the method on all pulsars.
    :param k_threshold: Threshold for dropout analysis.
    :param idx:
        Index of radio frequency dependence (i.e. DM is 2). Any float will work.
    """

    if orf_bins is None:
        orf_bin_size = 7
    else:
        orf_bin_size = len(orf_bins)-1
    orfs = {'crn': None, 'crn_chrom': None, 'hd': model_orfs.hd_orf(),
            'gw_monopole': model_orfs.gw_monopole_orf(),
            'gw_dipole': model_orfs.gw_dipole_orf(),
            'st': model_orfs.st_orf(),
            'gt': model_orfs.gt_orf(tau=parameter.Uniform(-1.5, 1.5)('tau')),
            'dipole': model_orfs.dipole_orf(),
            'monopole': model_orfs.monopole_orf(),
            'param_multiple_corr': model_orfs.param_multiple_corr_orf(
                mp=parameter.Uniform(0.0, 1.0)('gw_orf_param_mc_monopole'),
                dp=parameter.Uniform(0.0, 1.0)('gw_orf_param_mc_dipole'),
                hd=parameter.Uniform(0.0, 1.0)('gw_orf_param_mc_hd')),
            'zero_diag_param_multiple_corr': model_orfs.param_multiple_corr_orf(
                mp=parameter.Uniform(0.0, 1.0)('gw_orf_param_mc_monopole_zero_diag'),
                dp=parameter.Uniform(0.0, 1.0)('gw_orf_param_mc_dipole_zero_diag'),
                hd=parameter.Uniform(0.0, 1.0)('gw_orf_param_mc_hd_zero_diag'), diag=1e-20),
            'param_monopole': model_orfs.param_monopole_orf(c=parameter.Uniform(
                0.0, 1.0)('gw_orf_param_monopole')),
            'zero_diag_param_monopole': model_orfs.param_monopole_orf(c=parameter.Uniform(
                0.0, 1.0)('gw_orf_param_monopole_zero_diag'), diag=1e-20),
            'param_hd': model_orfs.param_hd_orf(
                a=parameter.Uniform(-1.5, 3.0)('gw_orf_param_hd_0'),
                b=parameter.Uniform(-1.0, 0.5)('gw_orf_param_hd_1'),
                c=parameter.Uniform(-1.0, 1.0)('gw_orf_param_hd_2')),
            'zero_diag_param_hd': model_orfs.param_hd_orf(
                a=parameter.Uniform(-1.5, 3.0)('gw_orf_param_hd_zero_diag_0'),
                b=parameter.Uniform(-1.0, 0.5)('gw_orf_param_hd_zero_diag_1'),
                c=parameter.Uniform(-1.0, 1.0)('gw_orf_param_hd_zero_diag_2'), diag=1e-20),
            'spline_orf': model_orfs.spline_orf(params=parameter.Uniform(
                -0.9, 0.9, size=7)('gw_orf_spline')),
            'interp_orf': model_orfs.interp_orf(params=parameter.Uniform(
                -1.0, 1.0, size=7)('gw_orf_interp')),
            'bin_orf': model_orfs.bin_orf(params=parameter.Uniform(
                -1.0, 1.0, size=orf_bin_size)('gw_orf_bin'), bins=orf_bins),
            'bin_cos_orf': model_orfs.bin_cos_orf(params=parameter.Uniform(
                -1.0, 1.0, size=7)('gw_orf_bin_cos')),
            'freq_hd': model_orfs.freq_hd(params=[components, orf_ifreq]),
            'zero_diag_hd': model_orfs.hd_orf(diag=1e-20),
            'zero_diag_dipole': model_orfs.dipole_orf(diag=1e-20),
            'zero_diag_monopole': model_orfs.monopole_orf(diag=1e-20),
            'zero_diag_spline_orf': model_orfs.spline_orf(params=parameter.Uniform(
                -0.9, 0.9, size=7)('gw_orf_spline_zero_diag'), diag=1e-20),
            'zero_diag_interp_orf': model_orfs.interp_orf(params=parameter.Uniform(
                -1.0, 1.0, size=7)('gw_orf_interp_zero_diag'), diag=1e-20),
            'zero_diag_bin_orf': model_orfs.bin_orf(params=parameter.Uniform(
                -1.0, 1.0, size=7)('gw_orf_bin_zero_diag'), diag=1e-20),
            'zero_diag_bin_cos_orf': model_orfs.bin_cos_orf(params=parameter.Uniform(
                -1.0, 1.0, size=7)('gw_orf_bin_cos_zero_diag'), diag=1e-20),
            'legendre_orf': model_orfs.legendre_orf(params=parameter.Uniform(
                -1.0, 1.0, size=leg_lmax+1)('gw_orf_legendre')),
            'zero_diag_legendre_orf': model_orfs.legendre_orf(params=parameter.Uniform(
                -1.0, 1.0, size=leg_lmax+1)('gw_orf_legendre_zero_diag'), diag=1e-20),
            'chebyshev_orf': model_orfs.chebyshev_orf(params=parameter.Uniform(
                -1.0, 1.0, size=4)('gw_orf_chebyshev')),
            'zero_diag_chebyshev_orf': model_orfs.chebyshev_orf(params=parameter.Uniform(
                -1.0, 1.0, size=4)('gw_orf_chebyshev_zero_diag'), diag=1e-20)}

    if tnfreq and Tspan is not None:
        components = blocks.get_tncoeff(Tspan, components)

    # common red noise parameters
    if psd in ['powerlaw', 'turnover', 'turnover_knee',
               'broken_powerlaw','flat_powerlaw']:
        amp_name = '{}_log10_A'.format(name)
        if log10_A_val is not None:
            log10_Agw = parameter.Constant(log10_A_val)(amp_name)

        elif logmin is not None and logmax is not None:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(logmin, logmax)(amp_name)
            elif prior == 'log-uniform':
                log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
            elif prior == 'gaussian':
                log10_Agw = parameter.Normal(logmin, logmax)(amp_name)

        else:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
            elif prior == 'log-uniform':
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-20, -11)(amp_name)

        gam_name = '{}_gamma'.format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else: 
            if gamma_prior == 'uniform':
                gamma_gw = parameter.Uniform(gammamin, gammamax)(gam_name)
            elif gamma_prior == 'gaussian':
                gamma_gw = parameter.Normal(gammamin, gammamax)(gam_name)

        # common red noise PSD
        if psd == 'powerlaw':
            if any([dropout, dropbin]):
                if dropout:
                    if dropout_common:
                        k_drop = parameter.Uniform(0, 1)(name+"_k_drop")
                    else:
                        k_drop = parameter.Uniform(0, 1)
                else:
                    k_drop = None
                if dropbin:
                    if dropbin_common:
                        k_dropbin = parameter.Uniform(dropbin_min, components+1)(name+"_k_dropbin")
                    else:
                        k_dropbin = parameter.Uniform(dropbin_min, components+1)
                else:
                    k_dropbin = None
                
                cpl = drop.dropout_powerlaw(log10_A=log10_Agw, gamma=gamma_gw,
                                            dropout_psr=dropout_psr, k_drop=k_drop,
                                            dropbin_psr=dropbin_psr, k_dropbin=k_dropbin,
                                            k_threshold=k_threshold)
            else:
                cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == 'broken_powerlaw':
            delta_name = '{}_delta'.format(name)
            kappa_name = '{}_kappa'.format(name)
            log10_fb_name = '{}_log10_fb'.format(name)
            kappa_gw = parameter.Uniform(0.01, 0.5)(kappa_name)
            log10_fb_gw = parameter.Uniform(-10, -6)(log10_fb_name)

            if delta_val is not None:
                delta_gw = parameter.Constant(delta_val)(delta_name)
            else:
                delta_gw = parameter.Uniform(0, 7)(delta_name)
            cpl = gpp.broken_powerlaw(log10_A=log10_Agw,
                                      gamma=gamma_gw,
                                      delta=delta_gw,
                                      log10_fb=log10_fb_gw,
                                      kappa=kappa_gw)
        elif psd == 'turnover':
            kappa_name = '{}_kappa'.format(name)
            lf0_name = '{}_log10_fbend'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lf0_gw = parameter.Uniform(-9, -7)(lf0_name)
            cpl = utils.turnover(log10_A=log10_Agw, gamma=gamma_gw,
                                 lf0=lf0_gw, kappa=kappa_gw)
        elif psd == 'turnover_knee':
            kappa_name = '{}_kappa'.format(name)
            lfb_name = '{}_log10_fbend'.format(name)
            delta_name = '{}_delta'.format(name)
            lfk_name = '{}_log10_fknee'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lfb_gw = parameter.Uniform(-9.3, -8)(lfb_name)
            delta_gw = parameter.Uniform(-2, 0)(delta_name)
            lfk_gw = parameter.Uniform(-8, -7)(lfk_name)
            cpl = gpp.turnover_knee(log10_A=log10_Agw, gamma=gamma_gw,
                                   lfb=lfb_gw, lfk=lfk_gw,
                                   kappa=kappa_gw, delta=delta_gw)
        elif psd == 'flat_powerlaw':
            bmp_name = '{}_log10_B'.format(name)
            log10_Bgw = parameter.Uniform(-10, -4)(bmp_name)
            cpl = gpp.flat_powerlaw(log10_A=log10_Agw, gamma=gamma_gw,
                                   log10_B=log10_Bgw)

    if psd == 'spectrum':
        rho_name = '{}_log10_rho'.format(name)
        if logmin is not None and logmax is not None:
            if prior == 'uniform':
                log10_rho_gw = parameter.LinearExp(logmin, logmax,
                                                   size=components)(rho_name)
            elif prior == 'log-uniform':
                log10_rho_gw = parameter.Uniform(logmin, logmax,
                                                 size=components)(rho_name)
        else:
            if prior == 'uniform':
                log10_rho_gw = parameter.LinearExp(-10, -4,
                                                   size=components)(rho_name)
            elif prior == 'log-uniform':
                log10_rho_gw = parameter.Uniform(-10, -4,
                                                 size=components)(rho_name)
            else:
                log10_rho_gw = parameter.Uniform(-9, -4,
                                                 size=components)(rho_name)

        cpl = gpp.free_spectrum(log10_rho=log10_rho_gw)

    if select == 'backend':
        # define selection by observing backend
        selection = selections.Selection(selections.by_backend)
    elif select == 'band' or select == 'band+':
        # define selection by observing band
        selection = selections.Selection(selections.by_band)
    elif isinstance(select, list):
        # define selection by list of custom backend
        selection = selections.Selection(selections.custom_backends(select))
    elif isinstance(select, dict):
        # define selection by dict of custom backend
        selection = selections.Selection(selections.custom_backends_dict(select))
    elif isinstance(select, type):
        # define selection
        selection = select
    else:
        # define no selection
        selection = selections.Selection(selections.no_selection)

    # TODO: Find a way to use selection functions instead of flags. 
    # Note: Selecting here allows to select ToAs, even for correlated signals
    # (not available in enterprise currently)
    if flagval:
        cbasis = gpb.createfourierdesignmatrix_general(flagname=flagname, 
                                                       flagval=flagval,
                                                       idx=idx,
                                                       tndm=tndm,
                                                       nmodes=components,
                                                       Tspan=Tspan,
                                                       psrTspan=psrTspan,
                                                       logf=logf,
                                                       fmin=fmin,
                                                       fmax=fmax,
                                                       modes=modes,
                                                       pshift=pshift,
                                                       pseed=pseed)
    elif idx is not None:
        if tndm:
            cbasis = gpb.createfourierdesignmatrix_dm_tn(nmodes=components, Tspan=Tspan,
                                                         logf=logf, fmin=fmin, fmax=fmax,
                                                         idx=idx, modes=modes)
        else:
            cbasis = gpb.createfourierdesignmatrix_chromatic(nmodes=components, Tspan=Tspan,
                                                             logf=logf, fmin=fmin, fmax=fmax,
                                                             idx=idx, modes=modes)
    else:
        cbasis = gpb.createfourierdesignmatrix_red(nmodes=components, Tspan=Tspan,
                                                   logf=logf, fmin=fmin, fmax=fmax,
                                                   modes=modes, pshift=pshift, pseed=pseed)
    if orf is None or 'crn' in orf:
        crn = gp_signals.BasisGP(cpl, cbasis, coefficients=coefficients, combine=combine,
                                 selection=selection, name=name)
    elif orf in orfs.keys():
        crn = gp_signals.BasisCommonGP(cpl, cbasis, orfs[orf], coefficients=coefficients,
                                       combine=combine, name=name) 
    elif isinstance(orf, types.FunctionType):
        crn = gp_signals.BasisCommonGP(cpl, cbasis, orf, coefficients=coefficients,
                                       combine=combine, name=name) 
    else:
        raise ValueError('ORF {} not recognized'.format(orf))

    return crn



############ functions to recover the Fourier coeffs

def get_b(d, TNT, phiinv):
    Sigma_inv = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
    try:
        L = sl.cholesky(Sigma_inv, lower=True)
        mn = sl.cho_solve((L, True), d)
        test = 100
        coeffs = mn[:,None] + sl.solve_triangular(L, np.random.randn(L.shape[0],test), trans='T', lower=True)

    except np.linalg.LinAlgError:
        Q, R = sl.qr(Sigma_inv)
        Sigi = sl.solve(R, Q.T)
        mn = np.dot(Sigi, d)
        u, s, _ = sl.svd(Sigi)
        Li = u * np.sqrt(1/s)
        coeffs = mn[:,None] + np.dot(Li, np.random.randn(Li.shape[0],test))

    return coeffs.T

def reconstruct_coeffs(self, mlv=False, idx=None):
    """
    Parameters
    ----------
    mlv : bool
        Whether to use the maximum likelihood value for the reconstruction.

    idx : int, optional
        Index of the chain array to use.

    Returns
    -------
    wave : array
        A reconstruction of a single gaussian process signal realization.
    """

    if idx is None:
        idx = np.random.randint(self.burn, self.chain.shape[0])
    elif mlv:
        idx = self.mlv_idx

    # get parameter dictionary
    params = self.sample_posterior(idx)
    self.idx = idx
    coeffs = []

    TNrs, TNTs, phiinvs, Ts = self._get_matrices(params=params)

    for (p_ct, psrname, d, TNT, phiinv, T) in zip(self.p_idx, self.p_list,
                                                    TNrs, TNTs, phiinvs, Ts):
        coeffs.append(get_b(d, TNT, phiinv))

    return coeffs
