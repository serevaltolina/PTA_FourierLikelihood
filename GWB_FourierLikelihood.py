from __future__ import division

import sys, os, json, pickle
import numpy as np
import scipy.linalg as sl
import la_forge.gp as lfgp
import _pickle as cPickle

import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise_extensions import sampler

import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import sere_enterprise as sere


####################################   FUNCTIONS    #################################### 

# defining the PTA object and pulsars noise models
#--------------------------------------------------------------------------------

def pulsar_model(psr, model_base, inc_chrom=True, inc_dmdip=True, fix_dip=True, fix_chrom=True):

    # chromatic noise
    if psr.name == 'J1600-3053' and inc_chrom == True:
        
        if fix_chrom == True:
            chrom = sere.chromatic_noise_block(components=93,gp_kernel='diag', psd='flat_powerlaw',gamma_val=5, logA_val=-12,logk_val=-5)
        else:
            chrom = sere.chromatic_noise_block(components=93,gp_kernel='diag', psd='flat_powerlaw')
        
        model = model_base + chrom
        
    # DM dip
    elif psr.name == 'J1713+0747' and inc_dmdip == True:
        
        dm_expdip_tmin=[54650, 57490]
        dm_expdip_tmax=[54850, 57530]
        tmin = (dm_expdip_tmin if isinstance(dm_expdip_tmin, list)
                else [dm_expdip_tmin])
        tmax = (dm_expdip_tmax if isinstance(dm_expdip_tmax, list)
                else [dm_expdip_tmax])
        num_dmdips = 2
        dm_expdip_idx=[4,1] 
        dm_expdip_idx = (dm_expdip_idx if isinstance(dm_expdip_idx,list)
                                        else [dm_expdip_idx]*int(num_dmdips))
        
        dm_expdip_sign='negative'
        dm_expdip_sign = (dm_expdip_sign if isinstance(dm_expdip_sign,list)
                                            else [dm_expdip_sign]*int(num_dmdips))

        if fix_dip == True:
            for dd in range(num_dmdips):
                dmdip = sere.dm_exponential_dip(tmin=tmin[dd], tmax=tmax[dd],
                                                        idx=dm_expdip_idx[dd],
                                                        sign=dm_expdip_sign[dd],
                                                        name='dmexp_{0}'.format(dd+1),
                                                        t0_dmexp_val=57510, log10_Amp_dmexp_val=-6, log10_tau_dmexp_val=1.5)
        else:
            for dd in range(num_dmdips):
                dmdip = sere.dm_exponential_dip(tmin=tmin[dd], tmax=tmax[dd],
                                                        idx=dm_expdip_idx[dd],
                                                        sign=dm_expdip_sign[dd],
                                                        name='dmexp_{0}'.format(dd+1))
        model = model_base + dmdip

    else:
        model = model_base

    return model(psr)


def build_pta (psrs, fix_rn=True, fix_dm=True, fix_wn=True, logA_red=-15, gamma_red=4, logA_dm=-16, gamma_dm=3, log_kappa=-6, inc_chrom=True, inc_dmdip=True, fix_dip=True, fix_chrom=True, inc_crn=True, hd=True):

    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # timing model
    tm = gp_signals.MarginalizingTimingModel(use_svd=True)

    # white noise
    if fix_wn == True:
        efac = parameter.Constant()
        equad = parameter.Constant()
    else:
        efac = parameter.Uniform(0.1,5)
        equad = parameter.Uniform(-10,-5)
    ef = white_signals.MeasurementNoise(efac=efac, selection=Selection(selections.by_backend))
    eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=Selection(selections.by_backend))

    # red noise
    if fix_rn == True:
        rn_log10_A = parameter.Constant(logA_red)
        rn_gamma = parameter.Constant(gamma_red)
        rn_log10_kappa = parameter.Constant(log_kappa)
    else:
        rn_log10_A = parameter.Uniform(-18, -12)
        rn_log10_kappa = parameter.Uniform(-9,-4)
        rn_gamma = parameter.Uniform(0,7)
    rn_pl = sere.powerlaw_flat_tail(log10_A=rn_log10_A, gamma=rn_gamma, log10_kappa=rn_log10_kappa)
    rn = gp_signals.FourierBasisGP(spectrum=rn_pl, components=30, Tspan=Tspan)

    # DM variations
    if fix_dm == True:
        dm_log10_A = parameter.Constant(logA_dm)
        dm_gamma = parameter.Constant(gamma_dm) 
        dm_log10_kappa = parameter.Constant(log_kappa)
    else:
        dm_log10_A = parameter.Uniform(-18, -12)
        dm_gamma = parameter.Uniform(0,7)
        dm_log10_kappa = parameter.Uniform(-9,-4)
    pl_dm = sere.powerlaw_flat_tail(log10_A=dm_log10_A, gamma=dm_gamma, log10_kappa=dm_log10_kappa)
    dm_basis = utils.createfourierdesignmatrix_dm(nmodes=100)
    dm = gp_signals.BasisGP(priorFunction=pl_dm, basisFunction=dm_basis, name='dm')

    if hd == True:
        # HD correlated background
        crn = sere.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan, components=9,  logmin=-15.5, logmax=-13.5, gamma_val=None, name='gw',orf='hd')
    else:
        # common red noise signal (uncorrelated)
        crn = sere.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan, components=9,  logmin=-15.5, logmax=-13.5, gamma_val=None, name='crn')

    # base model
    model_base = tm + ef + eq + rn + dm

    if inc_crn == True:
        model_base += crn

    return signal_base.PTA([pulsar_model(psr,model_base,inc_chrom=inc_chrom, inc_dmdip=inc_dmdip, fix_dip=fix_dip, fix_chrom=fix_chrom) for psr in psrs])


# update covariance formula
#--------------------------------------------------------------------------------

def update_cov(Cs,ms,ns,npsrs):

    C_list = []
    n_mat = len(ns)

    for p in range(npsrs):
        C = 0
        m = 0
        for i in range(n_mat):
            C += (ns[i])*Cs[i][p][:][:] + ns[i]*ms[i][p][:].T*ms[i][p][:] 
            m += ns[i]*ms[i][p][:]
        m = m/np.sum(ns)
        C = C - np.sum(ns)*m.T*m
        C = C/(np.sum(ns))
        C_list.append(C)
        
    return C_list


##### FOURIER LIKELIHOOD
#--------------------------------------------------------------------------------

def log_likelihood_Fourier(xs):

    '''
        New Fourier likelihood for pta array
    '''
    params = xs if isinstance(xs, dict) else pta_fl.map_params(xs)

    phiinvs = pta_fl.get_phiinv(params, logdet=True) 

    loglike = 0
    loglike += -0.5 * np.sum([ell for ell in pta_fl.get_rNr_logdet(params)])
    loglike += sum(pta_fl.get_logsignalprior(params))

    if pta_fl._commonsignals:
        
        phiinv, logdet_phiinv = phiinvs
        
        Sigma_inv = Sigma_0_inv - phiinv_0 + phiinv
        Li, lower = sl.cho_factor(Sigma_inv, lower=True)
        
        Sigma = sl.cho_solve((Li,True), np.identity(len(Li)))
        expval = sl.cho_solve((Li,True), Si0_a_hat)
        logdet_sigma = np.sum(2 * np.log(np.diag(Li)))
        
        loglike += - 0.5 * logdet_phiinv
        loglike += 0.5 * (np.dot(Si0_a_hat, expval) - logdet_sigma)

    else:

        for Sigma_0_inv_p, pl_0, pl, Si0_a_hat_p in zip(Sigma_0_inv,phiinvs_0, phiinvs, Si0_a_hat):

            phiinv, logdet_phiinv = pl
            phiinv_0, logdet_phiinv_0 = pl_0
            
            Sigma_inv = Sigma_0_inv_p - np.diag(phiinv_0) + np.diag(phiinv)
            Li, lower = sl.cho_factor(Sigma_inv, lower=True)

            Sigma = sl.cho_solve((Li,True), np.identity(len(Li)))
            expval = sl.cho_solve((Li,True), Si0_a_hat_p)
            logdet_sigma = np.sum(2 * np.log(np.diag(Li)))

            loglike += - 0.5 * logdet_phiinv
            loglike += 0.5 * (np.dot(Si0_a_hat_p, expval) - logdet_sigma)
    
    return loglike


##### sample jump proposal functions
#--------------------------------------------------------------------------------

# RN prior jump function factory
def jump_prior_rn(psr,pta):
    def function(x, iter, beta):
        q = x.copy()
                
        idx = [i for i, name in enumerate(pta.param_names) if psr.name+'_red_noise' in name]
        q_full = [pta.params[i].sample() for i in idx]
        q[idx] = q_full
                            
        x_logp = pta.get_lnprior(x)
        q_logp = pta.get_lnprior(q)
                    
        lqxy = x_logp - q_logp
                    
        return q, float(lqxy)

    function.__name__ = str(psr.name[:5])+'_rn'
    return function

# DM prior jump function factory
def jump_prior_dm(psr,pta):
    def function(x, iter, beta):
        q = x.copy()
                
        idx = [i for i, name in enumerate(pta.param_names) if psr.name+'_dm' in name]
        q_full = [pta.params[i].sample() for i in idx]
        q[idx] = q_full
                            
        x_logp = pta.get_lnprior(x)
        q_logp = pta.get_lnprior(q)
                    
        lqxy = x_logp - q_logp
                    
        return q, float(lqxy)

    function.__name__ = str(psr.name[:5])+'_dm'
    return function


# crn prior jump function
def draw_from_prior_crn(x, iter, beta):
    """
    Prior draw crn noise parameters
                
    The function signature is specific to PTMCMCSampler.
    """
                
    q = x.copy()
                
    idx = [i for i, name in enumerate(pta_fl.param_names) if 'crn_' in name]
    q_full = [pta_fl.params[i].sample() for i in idx]
    q[idx] = q_full
                        
    x_logp = pta_fl.get_lnprior(x)
    q_logp = pta_fl.get_lnprior(q)
                
    lqxy = x_logp - q_logp
                
    return q, float(lqxy)

####################################   MAIN    ##########################################

home_dir = '...'
mcmc_dir = home_dir + '...'
os.system('mkdir '+mcmc_dir)

i_logA = -12
i_gamma = 5
i_logk = -5

# reading the psrs pickle
pkl_dir = home_dir + 'psrsDR2new.pkl'
with open(pkl_dir, 'rb') as f:
    psrs = pickle.load(f)
    f.close()

Npsrs = len(psrs)
print(len(psrs))

# reading the WN dictionary
wn_dict_dir = home_dir + 'DR2new_Wnoise.json'
with open(wn_dict_dir, 'r') as f:
    Wnoise_ml = json.load(f)
    f.close()

# building the PTA objects
pta_0 = build_pta(psrs, fix_wn=False, fix_dip=False, inc_crn=False, inc_dmdip=True, inc_chrom=True)
pta_fl = build_pta(psrs, fix_rn=False, fix_dm=False, fix_chrom=False, inc_crn=True, fix_dip=True, inc_dmdip=True, inc_chrom=True)
pta_fl.set_default_params(Wnoise_ml)

# spna chains (step 1)
total_WNchain = []
check = 0
for p in psrs:
    
    chain_dir = home_dir + 'SPNA_' +p.name+'/'+p.name+'_DR2new/wn_Tspan10/chain_1.txt'
    psr_chain = np.loadtxt(chain_dir)
    inv_burn = int(0.25 * psr_chain.shape[0]) 
    if check == 0:
        total_WNchain = psr_chain[-inv_burn:,:-4]
        check = 1
    else:
        total_WNchain = np.concatenate((total_WNchain,psr_chain[-inv_burn:,:-4]),axis=1)
    print(total_WNchain.shape)

print('WN chains read')

print(len(pta_0.param_names))

# phiinv_0
phiinvs_0 = pta_0.get_phiinv([],logdet=True)

# Sigma_0_inv and a_hat
N = 10000 
ns = np.ones(N)
lfrec = lfgp.Signal_Reconstruction(psrs, pta_0, total_WNchain, burn=0)
lfrec.reconstruct_coeffs = sere.reconstruct_coeffs
Sigma_0_inv_list = []
a_hat_list = []

for i in range(N):

    Sigma_0_inv_list_step = []
    a_hat_list_step = []
    idx = np.random.choice(np.arange(len(total_WNchain)), size=1, replace=False)
    params = lfrec.sample_posterior(idx)
    TNTs = pta_0.get_TNT(params)
    TNrs = pta_0.get_TNr(params)

    
    for TNr, TNT, pl in zip(TNrs, TNTs, phiinvs_0):
        if TNr is None:
            continue

        phiinv, logdet_phi = pl
        Sigma_0_inv_step = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        Li, lower = sl.cho_factor(Sigma_0_inv_step, lower=True)
        a_hat_step = sl.cho_solve((Li, True), TNr)
        Sigma_0_inv_list_step.append(Sigma_0_inv_step)
        a_hat_list_step.append(a_hat_step)

    Sigma_0_inv_list.append(Sigma_0_inv_list_step)
    a_hat_list.append(a_hat_list_step)


Sigma_0_inv = update_cov(Sigma_0_inv_list,a_hat_list,ns,Npsrs)
a_hat = [] 
for p in range(Npsrs):
    temp = 0
    for i in range(N):
        temp += a_hat_list[i][p][:]
    a_hat.append(temp/N)

logdet_sigma_0 = 0
for S0 in Sigma_0_inv:
    Li, lower = sl.cho_factor(S0, lower=True)
    logdet_sigma_0 += np.sum(2 * np.log(np.diag(Li)))

# Sigma_0 * a hat
Si0_a_hat = []
for p in range(Npsrs):
    Si0_a_hat.append(np.dot(Sigma_0_inv[p][:][:], a_hat[p][:]))


if pta_fl._commonsignals:

    # phiinv_0
    phiinv_0 = []
    for i in range(Npsrs):
        phiinv_0 = np.concatenate((phiinv_0, phiinvs_0[i][0]))
    phiinv_0 = np.diag(phiinv_0)
    phiinv_00 = phiinv_0

    # Sigma_0_inv
    Sigma_0_inv = sl.block_diag(*Sigma_0_inv)

    # Sigma_0 * a hat
    Si0_a_hat_all = Si0_a_hat 
    Si0_a_hat = []
    for i in range(Npsrs):
        Si0_a_hat  = np.concatenate((Si0_a_hat , Si0_a_hat_all[i]))  


# pickle up a backup for the 0 quantitities
pkl_dir = mcmc_dir + '/Sigma_0_inv.pkl'
with open(pkl_dir, "wb") as output_file:
    cPickle.dump(Sigma_0_inv, output_file)

pkl_dir = mcmc_dir + '/a_hat.pkl'
with open(pkl_dir, "wb") as output_file2:
    cPickle.dump(a_hat, output_file2)

print('the 0 quantities are computed and pickled!')

# sampling params
np.savetxt(mcmc_dir+'/params.txt', pta_fl.param_names, fmt='%s')

# sampling
xs = {par.name: par.sample() for par in pta_fl.params}
print('test ln_likelihood: ',log_likelihood_Fourier(xs),'\n')

ndim = len(xs)
cov = np.diag(np.ones(ndim) * 0.01**2)
groups = None
Num = 2e6

sampler = ptmcmc(ndim, log_likelihood_Fourier, pta_fl.get_lnprior, cov, groups=groups, outDir=mcmc_dir, resume=False, seed=1234)
for p in psrs:
    temp = jump_prior_rn(p,pta_fl)
    sampler.addProposalToCycle(temp, 5)
    temp = jump_prior_dm(p,pta_fl)
    sampler.addProposalToCycle(temp, 5)
sampler.addProposalToCycle(draw_from_prior_crn, 5)

x0 = np.hstack([p.sample() for p in pta_fl.params])
sampler.sample(x0, Num, SCAMweight=30, AMweight=15, DEweight=50)

#  posteriors
chain = np.loadtxt(mcmc_dir+'/chain_1.txt')
pars = sorted(xs.keys())
burn = int(0.25 * chain.shape[0]) 

exp_lev = np.array([1, 2, 3])
mp_array = 1 - np.exp(-0.5*exp_lev**2)

fig = corner.corner(chain[burn:,-6:-4], 30, labels=pars[-2:], levels=mp_array, color='darkorange', smooth=True);
plot_dir = mcmc_dir + '/gwb_fourier.png'
fig.savefig(plot_dir)