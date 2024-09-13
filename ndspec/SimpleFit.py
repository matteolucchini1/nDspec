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
        print(fit_report(self.fit_result,show_correl=False))
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
        ax1.set_xlabel("Frequency (Hz)")  
        
        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return           
        
    def plot_model(self,plot_data=True,plot_components=False,params=None,
                   units="fpower",residuals="delchi",return_plot=False):
        if params is None:
            model = self.eval_model(params=self.params)
        else:
            model = self.eval_model(params=params)

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
            
class Fit_TimeAvgSpectrum():
    def __init__(self):
        self.model = None
        self.model_params = None
        self.likelihood = None
        self.fit_result = None
        self.data = None
        self.data_err = None
        self.energs = None
        self.emin = None
        self.emax = None
        self.ebounds = None
        self.ewidths = None
        self.response = None
        pass

    #tbd: allow a mwl data setter too
    def set_data(self,data,response):
        bounds_lo, bounds_hi, counts, error, exposure = load_pha(data,response)
        self.response = response.rebin_channels(bounds_lo,bounds_hi)
        #this needs to keep track of the bin widths when i switch to 
        #integrating models  in each energy bin
        self.energs = 0.5*(self.response.energ_hi+self.response.energ_lo)
        self.energ_bounds = self.response.energ_hi-self.response.energ_lo
        self.emin = bounds_lo
        self.emax = bounds_hi
        self.ebounds = 0.5*(self.emax+self.emin)
        self.ewidths = bounds_hi - bounds_lo
        #this load the spectrum in units of counts/s/keV
        self.data = counts/exposure/self.ewidths
        self.data_err = error/exposure/self.ewidths
        #here we keep track of which channels are noticed/ignored, by default
        #all are noticed
        self.ebounds_mask = np.full((self.response.n_chans), True)
        #we also need to double load the data to store it in a hidden unmasked 
        #ranges variable this is required in case users want to ignore/re-notice  
        #energy formally we could call some of these directly e.g. from the 
        #resposne, but storing this way makes the following methods far more 
        #readable
        self._emin_unmasked = self.emin
        self._emax_unmasked = self.emax
        self._ebounds_unmasked = self.ebounds
        self._ewidths_unmasked = self.ewidths
        self._data_unmasked = self.data
        self._data_err_unmasked = self.data_err

    def ignore_energies(self,bound_lo,bound_hi):
        """
        Adjusts the data arrays stored such that they (and the fit)
        ignore selected channels  based on their energy bounds.

        Parameters
        ----------
        bound_lo : float
            Lower bound of ignored energy interval
        bound_hi : float,
            Higher bound of ignored energy interval     
        """
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Energy bounds must be floats or integers")
        
        #if bounds of channel lie in ignored energies, ignore channel
        self.ebounds_mask = ((self._emin_unmasked<bound_lo)|
                             (self._emax_unmasked>bound_hi))&self.ebounds_mask
        
        #take the unmasked arrays and keep only the bounds we want
        self.emin = np.extract(self.ebounds_mask,self._emin_unmasked)
        self.emax = np.extract(self.ebounds_mask,self._emax_unmasked)
        self.ebounds = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        self.ewidths = np.extract(self.ebounds_mask,self._ewidths_unmasked)
        self.data = np.extract(self.ebounds_mask,self._data_unmasked)
        self.data_err = np.extract(self.ebounds_mask,self._data_err_unmasked)
        
    def notice_energies(self,bound_lo,bound_hi):
        """
        Adjusts the data arrays stored such that they (and the fit)
        notice selected (previously ignore) channels  based on their energy 
        bounds.

        Parameters
        ----------
        bound_lo : float
            Lower bound of ignored energy interval
        bound_hi : float,
            Higher bound of ignored energy interval     
        """
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Energy bounds must be floats or integers")
        
        #if bounds of channel lie in noticed energies, noitce channel
        self.ebounds_mask = self.ebounds_mask|np.logical_not(
                            (self._emin_unmasked<bound_lo)|
                            (self._emax_unmasked>bound_hi))
        
        #take the unmasked arrays and keep only the bounds we want
        self.emin = np.extract(self.ebounds_mask,self._emin_unmasked)
        self.emax = np.extract(self.ebounds_mask,self._emax_unmasked)
        self.ebounds = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        self.ewidths = np.extract(self.ebounds_mask,self._ewidths_unmasked)
        self.data = np.extract(self.ebounds_mask,self._data_unmasked)
        self.data_err = np.extract(self.ebounds_mask,self._data_err_unmasked)    
    
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
    
    def eval_model(self,params=None,energ=None,fold=True):    
        #this is fucked if we evaulate on a different energy grid
        if energ is None:
            energ = self.energs
        if params is None:
            eval = self.model.eval(self.model_params,energ=energ)* \
                   self.energ_bounds
        else:
            eval = self.model.eval(params,energ=energ)*self.energ_bounds
        #by default we fold the model through the response when evaluating it
        if fold is True:
            eval = self.response.convolve_response(eval)  
        return eval

    def get_residuals(self,model,res_type):
        model = self.eval_model()
        model = np.extract(self.ebounds_mask,model)
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
            res, err = self.get_residuals(model,"delchi")
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

    def _spectrum_minimizer(self,params):
        if self.likelihood is None:
            model = self.eval_model(params,energ=self.energs)
            convolve = np.extract(self.ebounds_mask,model)
            residuals = (self.data-convolve)/self.data_err
        else:
            raise TypeError("custom likelihood not implemented yet")
        return residuals

    def fit_data(self,algorithm='leastsq'):
        self.fit_result = minimize(self._spectrum_minimizer,self.model_params,
                                   method=algorithm)
        print(fit_report(self.fit_result,show_correl=False))
        fit_params = self.fit_result.params
        self.set_params(fit_params)
        return

    #need to track the units stuff is defined in better
    def plot_data(self,units="data",return_plot=False):
        energies = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        xerror = 0.5*np.extract(self.ebounds_mask,self._ewidths_unmasked)   
        
        if units == "data":
            data = self.data
            yerror = self.data_err
            ylabel = "Folded counts/s/keV"
        elif units.count("unfold"):
            power = units.count("e")            
            data = self.response.unfold_response(self._data_unmasked)* \
                   self._ebounds_unmasked**power
            error = self.response.unfold_response(self._data_err_unmasked)* \
                    self._ebounds_unmasked**power  
            data = np.extract(self.ebounds_mask,data)
            yerror = np.extract(self.ebounds_mask,error)
            if power == 0:
                ylabel = "Counts/s/keV/cm$^{2}$"
            elif power == 1:
                ylabel = "Flux density (Counts/s/cm$^{2}$)"
            elif power == 2:
                ylabel = "Flux (keV/s/cm$^{2}$)"
            #with weird units, use a generic label
            else:
                ylabel == "keV^{}/s/keV/cm$^{2}$".format(str(power))
        
        fig, ((ax1)) = plt.subplots(1,1,figsize=(6.,4.5))   
        
        ax1.errorbar(energies,data,yerr=yerror,xerr=xerror,
                     linestyle='',marker='o')
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel("Energy (keV)")          
        ax1.set_xscale("log",base=10)
        ax1.set_yscale("log",base=10)
        
        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return 

    def plot_model(self,plot_data=True,plot_components=False,params=None,
                        units="data",residuals="delchi",return_plot=False):
        #tbd: fold/unfold flag, get residuals in folded space always 
        energies = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        xerror = 0.5*np.extract(self.ebounds_mask,self._ewidths_unmasked)       
        
        #first; get the model in the correct units
        model_fold = self.eval_model(params=params,energ=self.energs)
        #model_fold = self.response.convolve_response(model_prefold)
        if units == "data":   
            model = np.extract(self.ebounds_mask,model_fold)   
            ylabel = "Folded counts/s/keV"
        elif units.count("unfold"):
            power = units.count("e")
            #the reason for folding and hten unfolding the model is that 
            #this is actually what happens to the data when we 'unfold' it - 
            #by definition it has gone from the physical space to the detector space
            #during an observation, and then we (arbitrarily) define the operation of
            #unfolding it to bring it back.   
            model = self.response.unfold_response(model_fold)
            if power == 0:
                ylabel = "Counts/s/keV/cm$^{2}$"
            elif power == 1:
                ylabel = "Flux density (Counts/s/cm$^{2}$)"
            elif power == 2:
                ylabel = "Flux (keV/s/cm$^{2}$)"
            #with weird units, use a generic label
            else:
                ylabel == "keV^{}/s/keV/cm$^{2}$".format(str(power))  
            model = np.extract(self.ebounds_mask,model)
            model = model*self.ebounds**power

        #if we're also plotting data, get the data in the same units
        #as well as the residuals
        if plot_data is True:
            model_res,res_errors = self.get_residuals(self.model,residuals)
            if residuals == "delchi":
                reslabel = "$\\Delta\\chi$"
            else:
                reslabel = "Data/model"                
            if units == "data":
                data = self.data
                yerror = self.data_err
                ylabel = "Folded counts/s/keV"
            elif units.count("unfold"):        
                data = self.response.unfold_response(self._data_unmasked)*\
                       self._ebounds_unmasked**power
                error = self.response.unfold_response(self._data_err_unmasked)*\
                        self._ebounds_unmasked**power  
                data = np.extract(self.ebounds_mask,data)
                yerror = np.extract(self.ebounds_mask,error)
            
        if plot_data is False:
            fig, (ax1) = plt.subplots(1,1,figsize=(7.5,6.))   
        else:
            fig, (ax1,ax2) = plt.subplots(2,1,figsize=(7.5,7.5),
                                          sharex=True,
                                          gridspec_kw={'height_ratios': [2, 1]})

        if plot_data is True:
            ax1.errorbar(energies,data,yerr=yerror,xerr=xerror,
                         ls="",marker='o')
       
       
        ax1.plot(energies,model,lw=3,zorder=3)

        if plot_components is True:
            #we need to allocate a ModelResult object in order to retrieve the components
            comps = LM_result(model=self.model,params=self.model_params).eval_components(energ=self.energs)
            for key in comps.keys():
                comp_folded = self.response.convolve_response(comps[key])
                #do it better here
                if units == "data":   
                    comp = np.extract(self.ebounds_mask,comp_folded)
                    ax1.plot(energies,comp,label=key,lw=2)
                elif units.count("unfold"):
                    comp_unfold = self.response.unfold_response(comp_folded)
                    comp = np.extract(self.ebounds_mask,comp_unfold)
                    ax1.plot(energies,comp*energies**power,label=key,lw=2)
            ax1.legend(loc='best')
        
        ax1.set_xscale("log",base=10)
        ax1.set_yscale("log",base=10)
        if plot_data is False:
            ax1.set_xlabel("Energy (keV)")
        ax1.set_ylabel(ylabel)

        if plot_data is True:
            ax2.errorbar(energies,model_res,yerr=res_errors,
                         drawstyle="steps-mid",marker='o')
            if residuals == "delchi":
                ax2.plot(energies,np.zeros(len(energies)),ls=":",lw=2,color='black')
            elif residuals == "ratio":
                ax2.plot(energies,np.ones(len(energiess)),ls=":",lw=2,color='black')                
            ax2.set_xlabel("Energy (keV)")
            ax2.set_ylabel(reslabel)

        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return  


def load_pha(path,response):
    '''
    This function loads xray spectra with astropy
    '''
    with fits.open(path,filemap=False) as spectrum:
        extnames = np.array([h.name for h in spectrum])
        spectrum_data = spectrum['SPECTRUM'].data
        channels = spectrum_data['CHANNEL']
        counts = spectrum_data['COUNTS']
        #check if exposure is present in either the primary or spectrum headers
        try:
            exposure = spectrum['PRIMARY'].header['EXPOSURE']
        except KeyError:
             try:
                exposure = spectrum['SPECTRUM'].header['EXPOSURE']
             except KeyError:
                exposure = 1.
        #check if systematic errors are applied
        try: 
            sys_err = spectrum_data['SYS_ERR']   
            has_sys_err = True
        except KeyError:
            sys_err = np.zeros(len(counts))
            has_sys_err = False
        #check if the spectrum has been grouped
        try: 
            grouping_data = spectrum_data['GROUPING']  
            has_grouping = True
        except KeyError:
            has_grouping = False
        #calculate errors including systematics if present
        #note: we are summing the systematic and Poisson errors in quadrature
        #so the factor sqrt in the Poisson error factors out
        if has_sys_err:
            counts_err = np.sqrt(counts+np.power(counts*sys_err,2.))
        else:
            counts_err = np.sqrt(counts)
        #calculate the spectrum whether it has been grouped or not, along with 
        #the energy bounds, width, and errors for each bin in either case
        if has_grouping:
            group_start = np.where(grouping_data==1)[0]
            total_groups = len(group_start)
            counts_per_group = np.zeros(total_groups,dtype=int)
           # sys_err_per_group = np.zeros(total_groups)
            bin_bounds_lo = np.zeros(total_groups)
            bin_bounds_hi = np.zeros(total_groups)
            avg_sys = np.zeros(total_groups)
            for i in range(total_groups-1):
                counts_per_group[i] = np.sum(counts[group_start[i]:group_start[i+1]])
                avg_sys[i] = np.mean(sys_err[group_start[i]:group_start[i+1]])
                bin_bounds_lo[i] = response.emin[group_start[i]]
                #the upper bounds of this bin are the starting point of the next bin up in the grouping
                bin_bounds_hi[i] = response.emin[group_start[i+1]]    
            #the last bin needs to be accounted for explicitely because the photons may not end up
            #being regrouped
            counts_per_group[-1] = np.sum(counts[group_start[total_groups-1]:])
            avg_sys[-1] = np.mean(sys_err[group_start[total_groups-1]:])
            bin_bounds_lo[-1] = bin_bounds_hi[-2]
            bin_bounds_hi[-1] = response.emax[-1]
            sys_err_per_group = counts_per_group*avg_sys
            spectrum_error = np.sqrt(np.power(sys_err_per_group,2)+counts_per_group)
        else:
            bin_bounds_lo = response.emin
            bin_bounds_hi = response.emax
            counts_per_group = counts
            spectrum_error = counts_err
        return bin_bounds_lo, bin_bounds_hi, counts_per_group, spectrum_error, exposure
        #bound_midpoint, bin_diff, counts_per_group, rebin_error contains the rebinned spectrum
                    
