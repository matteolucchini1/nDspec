import numpy as np
import warnings

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
from matplotlib.colors import TwoSlopeNorm
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
fi = 22
plt.rcParams.update({'font.size': fi-5})

colorscale = pl.cm.PuRd(np.linspace(0.,1.,5))

from lmfit import fit_report, minimize
from lmfit.model import ModelResult as LM_result

class Fit_Powerspectrum():
    def __init__(self):
        self.model = None
        self.model_params = None
        self.likelihood = None
        self.fit_result = None
        self.data = None
        self.data_err = None
        self.freqs = None 
        pass
    
    def set_data(self,data,data_err=None,data_grid=None):
        if data.__module__ == "stingray.powerspectrum":
            self.data = data.power
            self.data_err = data.power_err
            self.freqs = data.freq            
        else:
            self.data = data
            self.data_err = data_err
            self.freqs = data_grid

    def set_model(self,model,params=None):
        #this should be an lmfit model object
        self.model = model 
        if params is None:
            self.model_params = self.model.make_params(verbose=True)
        else:
            self.model_params = params

    def set_params(self,params):
        #not sure this makes sense
        self.model_params = params
    
    def eval_model(self,params=None,freq=None):
        #this is a reasonable start for the wrapper in the future, may also be 
        #worth to build this into the minimizer
        if freq is None:
            freq = self.freqs
        if params is None:
            eval = self.model.eval(self.model_params,freq=freq)
        else:
            eval = self.model.eval(params,freq=freq)
        return eval
         
    def get_residuals(self,model,res_type):
        if res_type == "ratio":
            residuals = self.data/model
            bars = self.data_err/model
        elif res_type == "delchi":
            residuals = (self.data-model)/self.data_err
            bars = np.ones(len(self.data))
        else:
            #eventually a better likelihood will need to go here
            print("can only return delta chi squared or ratio")
        return residuals, bars

    def print_fit_stat(self):
        if self.likelihood is None:
            res, err = self.get_residuals(self.model.eval(self.model_params,
                                                          freq=self.freqs),
                                                          "delchi")
            chi_squared = np.sum(np.power(res.reshape(len(self.data)),2))
            n_pars = len(self.model_params)
            dof = len(self.data) - n_pars
            reduced_chisquared = chi_squared/dof
            print("Goodness of fit metrics:")
            print("Chi squared" + "{0: <13}".format(" ") + str(chi_squared))
            print("Reduced chi squared" + "{0: <5}".format(" ") + str(reduced_chisquared))
            print("Data bins:" + "{0: <14}".format(" ") + str(len(self.data)))
            print("Free parameters:" + "{0: <8}".format(" ") + str(n_pars))
            print("Degrees of freedom:" + "{0: <5}".format(" ") + str(dof))
        else:
            print("custom likelihood not supported yet")
    
    def _psd_minimizer(self,params):
        if self.likelihood is None:
            model = self.model.eval(params,freq=self.freqs)
            residuals = (self.data-model)/self.data_err
        else:
            raise TypeError("custom likelihood not implemented yet")
        return residuals

    def fit_data(self,algorithm='leastsq'):
        self.fit_result = minimize(self._psd_minimizer,self.model_params,
                                   method=algorithm)
        print(fit_report(self.fit_result,min_correl=0.7))
        fit_params = self.fit_result.params
        self.set_params(fit_params)
        return
    
    def plot_data(self,units="fpower",return_plot=False):
        fig, ((ax1)) = plt.subplots(1,1,figsize=(6.,4.5))   
        
        if units == 'power':
            ax1.errorbar(self.freqs,self.data,
                         yerr=self.data_err,
                         drawstyle="steps-mid",marker='o')
            ax1.set_ylabel("Power")
        elif units == "fpower":
            ax1.errorbar(self.freqs,self.data*self.freqs,
                         yerr=self.data_err*self.freqs,
                         drawstyle="steps-mid",marker='o')
            ax1.set_ylabel("Power$\\times$frequency")
        else:
            raise ValueError("Y axis units not recognized")
        
        ax1.set_xscale("log",base=10)
        ax1.set_yscale("log",base=10)
        ax1.set_xlabel("Frequency")  
        
        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return           
        
    def plot_model(self,plot_data=True,plot_components=False,params=None,
                   units="fpower",residuals="delchi",return_plot=False):
        if params is None:
            model = self.eval_model(params=params)
        else:
            model = self.eval_model(params=self.params)

        if plot_data is True:
            model_res, res_errors = self.get_residuals(model,res_type=residuals)
            if residuals == "delchi":
                reslabel = "$\\Delta\\chi$"
            else:
                reslabel = "Data/model"
            if units == "power":
                data = self.data
                error = self.data_err
            elif units == "fpower":
                data = self.data*self.freqs
                error = self.data_err*self.freqs
                model = model*self.freqs

        
        if units == "power":
            ylabel = "Power"
        elif units == "fpower":
            ylabel= "Power$\\times$frequency"
            
        if plot_data is False:
            fig, (ax1) = plt.subplots(1,1,figsize=(6.,4.5))   
        else:
            fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6.,6.),
                                          sharex=True,
                                          gridspec_kw={'height_ratios': [2, 1]})

        if plot_data is True:
            ax1.errorbar(self.freqs,data,yerr=error,
                         drawstyle="steps-mid",marker='o')
       
       
        ax1.plot(self.freqs,model,lw=3,zorder=3)

        if plot_components is True:
            #we need to allocate a ModelResult object in order to retrieve the components
            comps = LM_result(model=self.model,params=test.model_params)
            comps = comps.eval_components(freq=test.freqs)
            for key in comps.keys():
                if units == "power":                
                    ax1.plot(test.freqs,comps[key],label=key,lw=2)
                elif units == "fpower":
                    ax1.plot(test.freqs,comps[key]*test.freqs,label=key,lw=2)
            ax1.legend(loc='best')
        
        ax1.set_xscale("log",base=10)
        ax1.set_yscale("log",base=10)
        if plot_data is False:
            ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel(ylabel)
        ax1.set_ylim([0.3*np.min(model),3.*np.max(model)])

        if plot_data is True:
            ax2.errorbar(self.freqs,model_res,yerr=res_errors,
                         drawstyle="steps-mid",marker='o')
            if residuals == "delchi":
                ax2.plot(self.freqs,np.zeros(len(self.freqs)),
                         ls=":",lw=2,color='black')
            elif residuals == "ratio":
                ax2.plot(self.freqs,np.ones(len(self.freqs)),
                         ls=":",lw=2,color='black')                
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel(reslabel)

        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return   
