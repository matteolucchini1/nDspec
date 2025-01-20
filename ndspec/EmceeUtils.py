import numpy as np 
import corner
import copy
import math

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
from matplotlib.colors import TwoSlopeNorm
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
fi = 22
plt.rcParams.update({'font.size': fi-5})

names = None 
global_priors = None
emcee_data = None 
emcee_data_err = None
emcee_model = None 

def set_emcee_priors(priors):
    global global_priors
    global_priors = priors 
    return 

def set_emcee_model(model): 
    global emcee_model
    emcee_model = model  
    return 
    
def set_emcee_data(data,error):
    global emcee_data
    global emcee_data_err
    emcee_data = data 
    emcee_data_err = error 
    return     

def set_emcee_parameters(params):
    """
    asdf
    """
    global names 
    global values 
    global emcee_params
    
    emcee_params = copy.copy(params) 
    values = []
    names = []
    theta = []
    for key in params:
        names = np.append(names,params[key].name)
        values = np.append(values,params[key].value)
        if params[key].vary is True:
            theta = np.append(theta,params[key].value)  
    return theta
    
class priorUniform():
    
    def __init__(self,min,max):
        self.min = min
        self.max = max
        pass 
        
    def logprob(self,theta):
        if self.min < theta < self.max:
            return 0.0
        return -np.inf


class priorLogUniform():
    
    def __init__(self,min,max):
        self.min = min
        self.max = max
        pass
    
    def logprob(self,theta):
        if self.min < theta < self.max:
            return -np.log(theta)
        return -np.inf


class priorNormal():
    
    def __init__(self,sigma,mu):
        self.sigma = sigma
        self.mu = mu
        pass 

    def logprob(self,theta):
        logprior = -0.5*(theta-self.mu)**2/self.sigma**2-0.5*np.log(2.*np.pi*self.sigma**2)
        return logprior


class priorLogNormal():
    def __init__(sigma,mu):
        self.sigma = sigma
        self.mu = mu
        pass 

    def logprob(self,theta):
        logprior = -0.5*(np.log(theta)-mu)**2/sigma**2-0.5*np.log(2.*np.pi*sigma**2/theta**2)
        return logprior

        
def log_priors(theta, prior_dict):
    logprior = 0
    for (key, obj), val in zip(prior_dict.items(), theta):        
        logprior = logprior + obj.logprob(val) 
    return logprior

    
def chi_square_likelihood(theta):
    global global_priors
    global names 
    global emcee_params
    global emcee_data
    global emcee_data_err
    global emcee_model 
    
    logpriors = log_priors(theta, global_priors)
    if not np.isfinite(logpriors):
        return -np.inf        
    for name, val in zip(names, theta):
        emcee_params[name].value = val    
    model = emcee_model(params=emcee_params)
    residual = (emcee_data-model)/emcee_data_err
    statistic = -0.5*np.sum(residual**2)
    #we're avoiding the constant terms because they only matter for 
    #model comparison, which we won't bother with for now. 
    likelihood = statistic + logpriors
    return likelihood

#note: this appears to be horribly messed up for power spectra 
def poisson_likelihood(theta,obs_time):
    global global_priors
    global names 
    global emcee_params
    global emcee_data
    global emcee_model 

    logpriors = log_priors(theta, global_priors)
    if not np.isfinite(logpriors):
        return -np.inf       
    for name, val in zip(names, theta):
        emcee_params[name].value = val    
    model = emcee_model(params=emcee_params)
    cash = model - emcee_data/obs_time + emcee_data/obs_time*(np.log(emcee_data/obs_time)-np.log(model))
    statistic = -np.sum(cash)
    likelihood = statistic + logpriors
    return likelihood

#this isn't terrible for now? uh
def whittle_likelihood(theta,segments):
    global global_priors
    global names 
    global emcee_params
    global emcee_data
    global emcee_model 
    
    logpriors = -log_priors(theta, global_priors)
    if not np.isfinite(logpriors):
        return -np.inf       
    for name, val in zip(names, theta):
        emcee_params[name].value = val    
    model = emcee_model(params=emcee_params)
    nu = 2.*segments
    whittle = emcee_data/model + np.log(model) + (2./nu -1)*np.log(emcee_data)
    statistic = -nu*np.sum(whittle)
    likelihood = statistic + logpriors
    return likelihood
    
def process_emcee(sampler,labels=None,discard=2000,thin=15,values=None):
    #print auto correlation lengths 
    tau = sampler.get_autocorr_time()
    with np.printoptions(threshold=np.inf):
        print("Autocorrelation lengths: ",tau)
    #print trace plots
    ndim = len(tau)
    size = math.ceil(14/9*ndim)
    fig, axes = plt.subplots(ndim, figsize=(9, size), sharex=True)
    samples = sampler.get_chain()    
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        if labels is not None:
            ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)    
    axes[-1].set_xlabel("Step number");
    #print acceptance fraction
    frac = sampler.acceptance_fraction
    nwalkers = len(frac)    
    fig, ax = plt.subplots(1, figsize=(6, 4), sharex=True)
    ax.scatter(np.linspace(0,nwalkers,nwalkers), frac, marker='o', alpha=0.8,color='black')
    ax.set_xlim(-1, nwalkers+1)
    ax.set_ylabel("Acceptance fraction")
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylim(0.9*np.min(frac),1.1*np.max(frac))    
    ax.set_xlabel("Walker");
    #corner plot,values from user input if desired         
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    fig = corner.corner(flat_samples, labels=labels, truths=values);
    return
