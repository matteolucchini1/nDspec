import numpy as np 
import corner
import copy
import math

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
from matplotlib.colors import TwoSlopeNorm

from .JointFit import JointFit
from .SimpleFit import SimpleFit

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
fi = 22
plt.rcParams.update({'font.size': fi-5})

emcee_names = None 
emcee_values = None
emcee_priors = None
emcee_data = None 
emcee_data_err = None
emcee_model = None 

def set_emcee_priors(fitobj,priors):
    """
    This function is used to set the priors to be used with emcee sampling.  
    These priors are saved in a global variable called emcee_priors; therefore,  
    users should never re-use the variable name emcee_priors in their code.
    
    Parameters:
    -----------
    fitobj: ndspec.Fit...Object or ndspec.JointFit 
        Object containing the data, specified model, and parameters
    priors: dict 
        A dictionary of priors to be used in emcee. The key of each dictionary 
        should be the name of the parameter. Each key should contain an object 
        with a method called "logprob", which returns the (negative) logarithm 
        of the prior evaluated at a given point.
    """
    
    global emcee_priors
    input_par_names = list(priors.keys())
    obj_par_names = list(fitobj.model_params.keys())

    if set(input_par_names) > set(obj_par_names):
        raise ValueError("Not all specified priors are parameters present in the model")
    
    for key in obj_par_names:
        if (fitobj.model_params[key].vary is True) and (key not in input_par_names):
            raise ValueError(f"{key} does not have a prior. Fix {key} or specify one.")
        elif (fitobj.model_params[key].vary is False) and (key in input_par_names):
            raise ValueError("Incorrectly specified a prior for a fixed parameter")
        else:
            continue

    emcee_priors = priors 
    return 

def set_emcee_model(fitobj): 
    """
    This function is used to set the model to be used with emcee sampling.  
    This model is saved in a global variable called emcee_model; therefore,  
    users should never re-use the variable name emcee_model in their code.
    
    Parameters:
    -----------
    fitobj: ndspec.Fit...Object or ndspec.JointFit 
        Object containing the data, specified model, and parameters
    """
    
    global emcee_model
    if type(fitobj) == JointFit:
        fitobj.flatten = True
    emcee_model = fitobj.eval_model
    return 
    
def set_emcee_data(fitobj):
    """
    This function is used to set the data and its error to be used with emcee 
    sampling. These are saved in global variables called emcee_data and 
    emcee_data_err; therefore, users should never re-use the variable names 
    emcee_data and emcee_data_err in their code.
    
    Parameters:
    -----------
    fitobj: ndspec.Fit...Object or ndspec.JointFit 
        Object containing the data, specified model, and parameters
    """
    
    global emcee_data
    global emcee_data_err
    if type(fitobj) == JointFit:
        emcee_data = np.array([])
        emcee_data_err = np.array([])
        for obs in fitobj.joint:
            if type(fitobj.joint[obs]) == list:
                for m in fitobj.joint[obs]:
                    emcee_data = np.concat([emcee_data,m.data])
                    emcee_data_err = np.concat([emcee_data_err,m.data_err])
            else:
                emcee_data = np.concat([emcee_data,fitobj.joint[obs].data])
                emcee_data_err = np.concat([emcee_data_err,fitobj.joint[obs].data_err])
    else:
        emcee_data = fitobj.data 
        emcee_data_err = fitobj.data_err
    return

def set_emcee_parameters(params):
    """
    This function is used to set the parameters of the model to be used with
    emcee sampling. The parameter object (containing all parameters), as well as
    the names and values of the variable parameters are saved in global 
    variables called emcee_names, emcee_values and emcee_params; therefore, 
    users should never re-use these variable names in their code.
    
    Parameters:
    -----------
    params: lmfit.Parameters
        The lmfit parameters object used in the model, including those kept
        constant. 
        
    Returns:
    -------
    theta: np.array 
        A numpy array containing the values of the free parameters in the model.
    """
    
    global emcee_names 
    global emcee_values 
    global emcee_params
    
    emcee_params = copy.copy(params) 
    emcee_values = []
    emcee_names = []
    theta = []
    for key in params:
        if params[key].vary is True:
            emcee_names = np.append(emcee_names,params[key].name)
            emcee_values = np.append(emcee_values,params[key].value)
            theta = np.append(theta,params[key].value)  
    return theta

def initialise_mcmc(fitobj,priors):
    """
    This function is used to initialise an MCMC run. The Fit...Object can be
    any of the particular data products, or a JointFit object containing
    multiple FitObjects.

    Parameters:
    -----------
    fitobj: ndspec.Fit...Object or ndspec.JointFit 
        Object containing the data, specified model, and parameters

    priors: dict
        A dictionary of priors to be used in emcee. The key of each dictionary 
        should be the name of the parameter. Each key should contain an object 
        with a method called "logprob", which returns the (negative) logarithm 
        of the prior evaluated at a given point.

    Returns:
    -------
    theta: np.array 
        A numpy array containing the values of the free parameters in the model.
    """
    global emcee_priors
    global emcee_names 
    global emcee_params
    global emcee_data
    global emcee_data_err
    global emcee_model 

    if type(fitobj) == JointFit:
        pass
    elif issubclass(type(fitobj),SimpleFit):
        pass
    else:
        raise TypeError("Invalid fit object passed")
    
    theta = set_emcee_parameters(fitobj.model_params)
    set_emcee_data(fitobj)
    set_emcee_model(fitobj)
    set_emcee_priors(fitobj,priors)
    return theta

class priorUniform():
    """
    This class is used to compute a uniform prior distribution during Bayesian
    sampling, for a given model parameter. 
    
    Parameters:
    -----------
    min: float 
        The lower bound of the distribution. 
        
    max: float 
        The upper bound of the distribution. 
    """
    
    def __init__(self,min,max):
        self.min = min
        self.max = max
        pass 
        
    def logprob(self,theta):
        """
        This method returns the log probability of the distribution - in this 
        case, an (arbitrary, for the purpose of likelihood optimization) 
        constant. 
        
        Parameters:
        -----------
        theta: float 
            The parameter value for which the likelihood is to be computed. 
            
        Returns:
        --------
            The value of the likelihood for the input parameter.
        """
        
        if self.min < theta < self.max:
            return 0.0
        return -np.inf


class priorLogUniform():
    """
    This class is used to compute a log-uniform prior distribution (in base e)
    during Bayesian sampling, for a given model parameter.
    
    Parameters:
    -----------
    min: float 
        The lower bound of the distribution 
        
    max: float 
        The upper bound of the distribution 
    """    
    
    def __init__(self,min,max):
        self.min = min
        self.max = max
        pass
    
    def logprob(self,theta):
        """
        This method returns the log probability of the distribution - in this 
        case, an (arbitrary, for the purpose of likelihood optimization) 
        constant. More explicitely, if x is our parameter and log(x) is uniform,
        then p(log(x)) = const, p(x) = p(log(x))*dlog(x)/dx = const/x.
        Therefore, the log-probability is (minus a constant)
        log(p(x)) = log(1/x) = -log(x). 
        
        Parameters:
        -----------
        theta: float 
            The parameter value for which the likelihood is to be computed. 
            
        Returns:
        --------
            The value of the likelihood for the input parameter.
        """        
        
        if self.min < theta < self.max:
            return -np.log(theta)
        return -np.inf


class priorNormal():
    """
    This class is used to compute a normal prior distribution during Bayesian
    sampling, for a given model parameter. 
    
    Parameters:
    -----------
    sigma: float 
        The standard deviation of the distribution. 
        
    mu: float 
        The expectation of the distribution. 
    """    
    
    def __init__(self,sigma,mu):
        self.sigma = sigma
        self.mu = mu
        pass 

    def logprob(self,theta):
        """
        This method returns the log probability of the distribution for the 
        given parameter theta.
        
        Parameters:
        -----------
        theta: float 
            The parameter value for which the likelihood is to be computed. 
            
        Returns:
        --------
            The value of the likelihood for the input parameter.
        """
        
        logprior = -0.5*(theta-self.mu)**2/self.sigma**2+0.5*np.log(2.*np.pi*self.sigma**2)
        return logprior


class priorLogNormal():
    """
    This class is used to compute a lognormal prior distribution during Bayesian
    sampling, for a given model parameter. 
    
    Parameters:
    -----------
    sigma: float 
        The standard deviation of the distribution. 
        
    mu: float 
        The expectation of the distribution. 
    """    
    
    def __init__(self,sigma,mu):
        self.sigma = sigma
        self.mu = mu
        pass 

    def logprob(self,theta):
        """
        This method returns the log probability of the distribution for the 
        given parameter theta.
        
        Parameters:
        -----------
        theta: float 
            The parameter value for which the likelihood is to be computed. 
            
        Returns:
        --------
            The value of the likelihood for the input parameter.
        """
        logprior = -0.5*(np.log(theta)-self.mu)**2/self.sigma**2+0.5*np.log(2.*np.pi*self.sigma**2/theta**2)
        return logprior

        
def log_priors(theta, prior_dict):
    """
    This function computes the total log-probability of a set of priors, given 
    a st of input parameter values. 
    
    Parameters:
    -----------
    theta: np.array(float)
        An array of parameter values for which to compute the priors 
            
    prior_dict: dictionary
        A dictionary of prior objects, each containing a method called .logprob 
        which returns the log-probability given the input parameter value 
        
    Returns:
    --------
    logprior: float 
        A float containing the log-probability of the set of parameters, given 
        their priors. 
    """

    logprior = 0
    for (key, obj), val in zip(prior_dict.items(), theta):        
        logprior = logprior + obj.logprob(val) 
    return logprior

    
def chi_square_likelihood(theta):
    """
    This function computes the log-likelihood, using the chi-square statistic
    and including priors, for a given set of parameter values theta. It requires
    the user to have set the global variables emcee_priors, emcee_names, 
    emcee_params, emcee_data, emcee_data_err and emcee_model beforehand. 
    
    Parameters: 
    -----------
    theta: np.array(float)
        An array of parameter values for which to compute the log likelihood. 
        
    Returns:
    --------
    likelihood: float 
        The value of the chi-square log-likelihood for the given parameter 
        values.
    """    

    global emcee_priors
    global emcee_names 
    global emcee_params
    global emcee_data
    global emcee_data_err
    global emcee_model 
    
    logpriors = log_priors(theta, emcee_priors)
    if not np.isfinite(logpriors):
        return -np.inf        
    for name, val in zip(emcee_names, theta):
        emcee_params[name].value = val    
    model = emcee_model(params=emcee_params)
    residual = (emcee_data-model)/emcee_data_err
    statistic = -0.5*np.sum(residual**2)
    likelihood = statistic + logpriors
    return likelihood

#note: double check units/obs_time
#def cash_likelihood(theta,obs_time):
#    """
#    This function computes the log-likelihood, using the Cash statistic
#    and including priors, for a given set of parameter values theta. It requires
#    the user to have set the global variables emcee_priors, emcee_names, 
#    emcee_params, emcee_data, emcee_data_err and emcee_model beforehand. Here, 
#    the definition of the Cash likelihood is identical to that of Xspec,  
#    https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html.
    
#    Input: 
#    ------
#    theta: np.array(float)
#        An array of parameter values for which to compute the log likelihood. 
#        
#    obs_time: float 
#        The exposure time of the observation. 
        
#    Returns:
#    --------
#    likelihood: float 
#        The value of the cstat log-likelihood for the given parameter values.        
#    """  

#    global emcee_priors
#    global emcee_names 
#    global emcee_params
#    global emcee_data
#    global emcee_model 

#    logpriors = log_priors(theta, emcee_priors)
#    if not np.isfinite(logpriors):
#        return -np.inf       
#    for name, val in zip(emcee_names, theta):
#        emcee_params[name].value = val    
#    model = emcee_model(params=emcee_params)
#    cash = model - emcee_data/obs_time + emcee_data/obs_time*(np.log(emcee_data/obs_time)-np.log(model))
#    statistic = -np.sum(cash)
#    likelihood = statistic + logpriors
#    return likelihood

#def whittle_likelihood(theta,segments):
#    """
#    This function computes the log-likelihood, using the Whittle statistic
#    and including priors, for a given set of parameter values theta. This
#    appropariate for modelling power spectra of binned time series data, and is 
#    described in Barret and Vaughan (2012) and Bachetti and Huppenkothen (2023):
#    https://ui.adsabs.harvard.edu/abs/2012ApJ...746..131B/abstract
#    https://ui.adsabs.harvard.edu/abs/2022arXiv220907954B/abstract
    
#    Input: 
#    ------
#    theta: np.array(float)
#        An array of parameter values for which to compute the log likelihood. 
        
#    segments: int 
#        The number of segments used to average the data in the powerspectrum.
        
#    Returns:
#    --------
#    likelihood: float 
#        The value of the cstat log-likelihood for the given parameter values.        
#    """ 

#    global emcee_priors
#    global emcee_names 
#    global emcee_params
#    global emcee_data
#    global emcee_model 
    
#    logpriors = -log_priors(theta, emcee_priors)
#    if not np.isfinite(logpriors):
#        return -np.inf       
#    for name, val in zip(emcee_names, theta):
#        emcee_params[name].value = val    
#    model = emcee_model(params=emcee_params)
#    nu = 2.*segments
#    whittle = emcee_data/model + np.log(model) + (2./nu -1)*np.log(emcee_data)
#    statistic = -nu*np.sum(whittle)
#    likelihood = statistic + logpriors
#    return likelihood
    
def process_emcee(sampler,labels=None,discard=2000,thin=100,values=None,get_autocorr=True):
    """
    Given a sampler emcee EnsamleSampler object, this function calculates and 
    prints the autocorrelation length, and plots the trace plots of the walkers, 
    the acceptance fraction, and the corner plot for the posteriors. 
    
    This function is meant for a quick look at the output of a chain, rather 
    than for publication quality plots. All the plots produced by this function 
    have more customization options than the default ones used here.  
    
    Parameters:
    -----------
    sampler: emcee.EnsamleSampler
        The sampler from which to plot the data 
    
    labels: list(str) 
        A list of strings to use for naming the parameters in both the trace and
        corner plots 
        
    discard: int, default 2000
        The number of steps used to define the burn-in period 
        
    thin: int, default 100
        Use one every "thin" steps in the chain. Used to make plots clearer. 
        
    values: np.array(float), default None
        An array of parameter values used to show the best fit or "true" value 
        of each parameter in the corner plot. 
    """

    #print auto correlation lengths 
    if get_autocorr is True:
        tau = sampler.get_autocorr_time()
        with np.printoptions(threshold=np.inf):
            print("Autocorrelation lengths: ",tau)
    
    #print trace plots
    ndim = sampler.ndim
    size = math.ceil(14/9*ndim)
    fig, axes = plt.subplots(ndim, figsize=(9, size), sharex=True)
    samples = sampler.get_chain(discard=discard, thin=thin)    
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
