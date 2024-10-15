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
    """
    Least-chi squared fitter class for a powerspectrum, defined as the product 
    between a Fourier-transformed lightcurve and its complex conjugate. 
    
    Given an array of Fourier frequencies, a power spectrum, its error and a 
    model (defined in Fourier space), this class handles fitting internally 
    using the lmfit library.    
    
    Poisson noise in the data is not accounted for explicitely. Users should 
    either pass noise-subtracted data, or include a constant component in the 
    model definition (see below).   
        
    Attributes
    ----------
    model: lmfit.CompositeModel 
        A lmfit CompositeModel object, which contains a wrapper to the model 
        component(s) one wants to fit to the data. 
   
    model_params: lmfit.Parameters 
        A lmfit Parameters object, which contains the parameters for the model 
        components.
   
    likelihood: None
        Work in progress; currently the software defaults to chi squared 
        likelihood
   
    fit_result: lmfit.MinimizeResult
        A lmfit MinimizeResult, which stores the result (including best-fitting 
        parameter values, fit statistics etc) of a fit after it has been run.         
   
    data: np.array(float)
        An array of float containing the power spectrum to be fitted. Users can 
        choose whatever units they prefer (e.g. fractional vs absolute 
        normalization), as long as it is a power (rather than the more commonly
        plotted power*frequency). 
   
    data_err: np.array(float)
        The uncertainty on the power stored in data. 
   
    freqs: np.array(float)
        The Fourier frequency over which both the data and model are defined, 
        in units of Hz.           
    """ 

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
        """
        This method is used to pass the data users want to fit. The input can
        be either three arrays including the power, its erorr, and the Fourier 
        frequency grid, or a Stingray Powerspectrum object, in which case the 
        error and frequency array are handled automatically.
        
        Parameters:
        -----------           
        data: np.array(float) or stingray.powerspectrum
            The power spectrum to be fitted, either as a numpy array or a 
            stingray object 
            
        data_err: np.array(float), default: None 
            The error on the power spectrum. If passing a stingray object, this 
            is not necessary and is therefore ignored
            
        data_grid: np.array(float), default: None 
            The Fourier frequency grid over which the data (and model) are 
            defined. If passing a stingray object, this is not necessary and is 
            therefore ignored      
        """
        
        if data.__module__ == "stingray.powerspectrum":
            self.data = data.power
            self.data_err = data.power_err
            self.freqs = data.freq            
        else:
            self.data = data
            self.data_err = data_err
            self.freqs = data_grid
        return

    def set_model(self,model,params=None):
        """
        This method is used to pass the model users want to fit to the data. 
        Optionally it is also possible to pass the initial parameter values of 
        the model. 
        
        Parameters:
        -----------            
        model: lmfit.CompositeModel 
            The lmfit wrapper of the model one wants to fit to the data. 
            
        params: lmfit.Parameters, default: None 
            The parameter values from which to start evalauting the model during
            the fit. If it is not provided, all model parameters will default 
            to 0, set to be free, and have no minimum or maximum bound. 
        """
    
        #this should be an lmfit model object
        self.model = model 
        if params is None:
            self.model_params = self.model.make_params(verbose=True)
        else:
            self.model_params = params
        return 

    def set_params(self,params):
        """
        This method is used to set the model parameter names and values. It can
        be used both to initialize a fit, and to test different parameter values 
        before actually running the minimization algorithm.
        
        Parameters:
        -----------                       
        params: lmfit.Parameters
            The parameter values from which to start evalauting the model during
            the fit.  
        """
        
        self.model_params = params
        return 
    
    def eval_model(self,params=None,freq=None):
        """
        This method is used to evaluate and return the model values for a given 
        set of parameters,  over a given Fourier frequency grid. By default it  
        will evaluate the model over the data Fourier frequency grid and using  
        the values stored internally in the model_params attribute, but passing 
        a different grid/set of parameters is also possible.         
        
        Parameters:
        -----------                         
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are 
            provided, the model_params attribute is used.
            
        freq: np.array(float), default None
            The the Fourier frequencies over which to evaluted the model. If 
            none are provided, the same frequencies over which the data is 
            defined are used. 
            
        Returns:
        --------
        model: np.array(float)
            The model evaluated over the given Fourier frequency array, for the 
            given input parameters.   
        """
        
        if freq is None:
            freq = self.freqs
        if params is None:
            model = self.model.eval(self.model_params,freq=freq)
        else:
            model = self.model.eval(params,freq=freq)
        return model
         
    def get_residuals(self,model_vals,res_type):
        """
        This methods return the residuals (either as data/model, or as 
        contribution to the total chi squared) of the input model, given the 
        parameters set in model_parameters, with respect to the data. 
        
        Parameters:
        -----------
        model_vals: np.array(float)
            An array of model values to be compared against the data.
            
        res_type: string 
            If set to "ratio", the method returns the residuals defined as 
            data/model. If set to "delchi", it returns the contribution of 
            each Frequency bin to the total chi squared.
            
        Returns:
        --------
        residuals: np.array(float)
            An array of the same size as the data, containing the model 
            residuals in each bin.
            
        bars: np.array(float)
            An array of the same size as the residuals, containing the one sigma 
            range for each contribution to the residuals.           
        """
        
        if res_type == "ratio":
            residuals = self.data/model_vals
            bars = self.data_err/model_vals
        elif res_type == "delchi":
            residuals = (self.data-model_vals)/self.data_err
            bars = np.ones(len(self.data))
        else:
            #eventually a better likelihood will need to go here
            print("can only return delta chi squared or ratio")
        return residuals, bars

    def print_fit_stat(self):
        """
        This method compares the model defined by the user, using the last set
        of parameters to have been set in the class, to the data stored. It then
        prints the chi-squared goodness-of-fit to terminal, along with the 
        number of data bins, free parameters and degrees of freedom. 
        """
    
        if self.likelihood is None:
            res, err = self.get_residuals(self.model,"delchi")
            chi_squared = np.sum(np.power(res.reshape(len(self.data)),2))
            freepars = 0
            for key, value in self.model_params.items():
                param = self.model_params[key]
                if param.vary is True:
                    freepars += 1
            dof = len(self.data) - freepars
            reduced_chisquared = chi_squared/dof
            print("Goodness of fit metrics:")
            print("Chi squared" + "{0: <13}".format(" ") + str(chi_squared))
            print("Reduced chi squared" + "{0: <5}".format(" ") + str(reduced_chisquared))
            print("Data bins:" + "{0: <14}".format(" ") + str(len(self.data)))
            print("Free parameters:" + "{0: <8}".format(" ") + str(freepars))
            print("Degrees of freedom:" + "{0: <5}".format(" ") + str(dof))
        else:
            print("custom likelihood not supported yet")
        return 
    
    def _psd_minimizer(self,params):
        """
        This method is used exclusively when running a minimization algorithm.
        It evaluates the model for an input set of parameters, and then returns 
        the residuals in units of contribution to the total chi squared 
        statistic.
        
        Parameters:
        -----------                         
        params: lmfit.Parameters
            The parameter values to use in evaluating the model. These will vary 
            as the fit runs.
            
        Returns:
        --------
        residuals: np.array(float)
            An array of the same size as the data, containing the model 
            residuals in each bin.            
        """
        
        if self.likelihood is None:
            model = self.model.eval(params,freq=self.freqs)
            residuals = (self.data-model)/self.data_err
        else:
            raise TypeError("custom likelihood not implemented yet")
        return residuals

    def fit_data(self,algorithm='leastsq'):
        """
        This method attempts to minimize the residuals of the model with respect 
        to the data defined by the user. The fit always starts from the set of 
        parameters defined with .set_params(). Once the algorithm has completed 
        its run, it prints to terminal the best-fitting parameters, fit 
        statistics, and simple selection criteria (reduced chi-squared, Akaike
        information criterion, and Bayesian informatino criterion). 
        
        Parameters:
        -----------
        algorithm: str, default="leastsq"
            The fitting algorithm to be used in the minimization. The possible 
            choices are detailed on the LMFit documentation page:
            https://lmfit.github.io/lmfit-py/fitting.html#fit-methods-table.
        """
        
        self.fit_result = minimize(self._psd_minimizer,self.model_params,
                                   method=algorithm)
        print(fit_report(self.fit_result,show_correl=False))
        fit_params = self.fit_result.params
        self.set_params(fit_params)
        return
    
    def plot_data(self,units="fpower",return_plot=False):
        """
        This method plots the powerspectrum loaded by the user as a function of 
        Fourier frequency. It is possible to plot both units of power, and power 
        times frequency, depending on user input. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        units: str, default="fpower"
            The units to use for the y axis. units="fpower", the default, plots 
            the data in units of power*frequency. units="power" instead plots 
            the data in units of power. 
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        
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
        """
        This method plots the model defined by the user as a function of 
        Fourier frequency, as well as (optionally) its components, and the data
        plus model residuals. It is possible to plot both units of power, and 
        power times frequency, depending on user input. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        plot_data: bool, default=True
            If true, both model and data are plotted; if false, just the model. 
            
        plot_components: bool, default=False 
            If true, the model components are overplotted; if false, they are 
            not. Only additive model components will display their values 
            correctly. 
            
        params: lmfit.parameters, default=None 
            The parameters to be used to evaluate the model. If False, the set 
            of parameters stored in the class is used 
        
        units: str, default="fpower"
            The units to use for the y axis. units="fpower", the default, plots 
            the data in units of power*frequency. units="power" instead plots 
            the data in units of power. 
            
        residuals: str, default="delchi"
            The units to use for the residuals. If residuals="delchi", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        
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
    """
    Least-chi squared fitter class for a time averaged spectrum, defined as the  
    count rate as a function of photon channel energy bound. 
    
    Given an instrument response, a count rate spectrum, its error and a 
    model (defined in energy space), this class handles fitting internally 
    using the lmfit library.    
        
    Attributes
    ----------
    model: lmfit.CompositeModel 
        A lmfit CompositeModel object, which contains a wrapper to the model 
        component(s) one wants to fit to the data. 
   
    model_params: lmfit.Parameters 
        A lmfit Parameters object, which contains the parameters for the model 
        components.
   
    likelihood: None
        Work in progress; currently the software defaults to chi squared 
        likelihood
   
    fit_result: lmfit.MinimizeResult
        A lmfit MinimizeResult, which stores the result (including best-fitting 
        parameter values, fit statistics etc) of a fit after it has been run.         
   
    data: np.array(float)
        An array of float containing the time-averaged spectrum to be fitted. It 
        should be defined in detector space, in units of counts/s/keV. Contains 
        exclusively the energy channels noticed during the fit.
   
    data_err: np.array(float)
        The uncertainty on the spectrum stored in data. Contains exclusively the
        energy channels noticed during the fit.

    response: nDspec.ResponseMatrix
        The instrument response matrix corresponding to the spectrum to be 
        fitted. It is required to define the energy grids over which model and
        data are defined. 
   
    energs: np.array(float)
        The array of physical photon energies over which the model is computed. 
        Defined as the middle of each bin in the energy range stored in the 
        instrument response provided.    
        
    energ_bounds: np.array(float)
        The array of energy bin widths, for each bin over which the model is 
        computed. Defined as the difference between the uppoer and lower bounds 
        of the energy bins stored in the insrument response provided. 
        
    emin: np.array(float)
        The array of lower bounds for the instrument energy channels, as stored 
        in the instrument response provided. Only contains the channels that are 
        noticed during the fit.
        
    emax: np.array(float)
        The array of upper bounds for the instrument energy channels, as stored 
        in the instrument response provided. Only contains the channels that are 
        noticed during the fit.
        
    ebounds: np.array(float) 
        The array of energy channel bin centers for the instrument energy
        channels,  as stored in the instrument response provided. Only contains 
        the channels that are noticed during the fit.

    ewidths: np.array(float) 
        The array of energy channel bin widths for the instrument energy
        channels,  as stored in the instrument response provided. Only contains 
        the channels that are noticed during the fit.
        
    ebounds_mask: np.array(bool)
        The array of instrument energy channels that are either ignored or 
        noticed during the fit. A given channel i is noticed if ebounds_mask[i]
        is True, and ignored if it is false.
        
    _emin_unmasked, _emax_unmasked, _ebounds_unmasked, _ewidths_unmasked: np.array(float)
        The array of every lower bound, upper bound, channel center and channel 
        widths stored in the response, regardless of which ones are ignored or 
        noticed during the fit. Used exclusively to facilitate book-keeping 
        internal to the class. 
        
    _data_unmasked, _data_err_unmasked: np.array(float)
        The array of every cout rate and relative error contained in the 
        spectrum, regardless of which ones are ignored or noticed during the 
        fit. Used exclusively to facilitate book-keeping internal to the class. 
    """ 

    def __init__(self):
        self.model = None
        self.model_params = None
        self.likelihood = None
        self.fit_result = None
        self.data = None
        self.data_err = None
        self.response = None
        self.energs = None
        self.ewidths = None
        self.emin = None
        self.emax = None
        self.ebounds = None
        pass

    def set_data(self,response,data):
        """
        This method sets the data to be fitted, its error, and the  energy and 
        channel grids given an input spectrum and its associated response matrix. 
        
        If the file provided was grouped with heatools, the method loads the 
        grouped data and adjusts the channel grid automatically. The data is 
        assumed to be background-subtracted (or to have negligible background).
        
        Parameters:
        -----------
        response: nDspec.ResponseMatrix
            An instrument response (including both rmf and arf) loaded into a 
            nDspec ResponseMatrix object. 
        
        data: str 
            A string pointing to the path of an X-ray spectrum file, stored in 
            a type 1 OGIP-formatted file (such as a pha file produced by a
            typical instrument reduction pipeline).
        """
    
        bounds_lo, bounds_hi, counts, error, exposure = load_pha(data,response)
        self.response = response.rebin_channels(bounds_lo,bounds_hi)
        #this needs to keep track of the bin widths when i switch to integrating models 
        #in each energy bin
        self.energs = 0.5*(self.response.energ_hi+self.response.energ_lo)
        self.energ_bounds = self.response.energ_hi-self.response.energ_lo
        self.emin = bounds_lo
        self.emax = bounds_hi
        self.ebounds = 0.5*(self.emax+self.emin)
        self.ewidths = bounds_hi - bounds_lo
        #this loads the spectrum in units of counts/s/keV
        self.data = counts/exposure/self.ewidths
        self.data_err = error/exposure/self.ewidths
        #here we keep track of which channels are noticed/ignored, by default
        #all are noticed
        self.ebounds_mask = np.full((self.response.n_chans), True)
        self._emin_unmasked = self.emin
        self._emax_unmasked = self.emax
        self._ebounds_unmasked = self.ebounds
        self._ewidths_unmasked = self.ewidths
        self._data_unmasked = self.data
        self._data_err_unmasked = self.data_err
        return 

    def ignore_energies(self,bound_lo,bound_hi):
        """
        Adjusts the data arrays stored such that they (and the fit) ignore 
        selected channels based on their energy bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored energy interval.
        bound_hi : float
            Higher bound of ignored energy interval .    
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
        return 
        
    def notice_energies(self,bound_lo,bound_hi):
        """
        Adjusts the data arrays stored such that they (and the fit) notice
        selected (previously ignore) channels  based on their energy 
        bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored energy interval.
        bound_hi : float,
            Higher bound of ignored energy interval.     
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
        return 
    
    def set_model(self,model,params=None):
        """
        This method is used to pass the model users want to fit to the data. 
        Optionally it is also possible to pass the initial parameter values of 
        the model. 
        
        Parameters:
        -----------            
        model: lmfit.CompositeModel 
            The lmfit wrapper of the model one wants to fit to the data. The 
            output of the model must be in units of photons/keV/cm^2/s.            
            
        params: lmfit.Parameters, default: None 
            The parameter values from which to start evalauting the model during
            the fit. If it is not provided, all model parameters will default 
            to 0, set to be free, and have no minimum or maximum bound. 
        """
        
        self.model = model 
        if params is None:
            self.model_params = self.model.make_params(verbose=True)
        else:
            self.model_params = params
        return 

    def set_params(self,params):
        """
        This method is used to set the model parameter names and values. It can
        be used both to initialize a fit, and to test different parameter values 
        before actually running the minimization algorithm.
        
        Parameters:
        -----------                       
        params: lmfit.Parameters
            The parameter values from which to start evalauting the model during
            the fit.  
        """
        
        self.model_params = params
        return 
    
    def eval_model(self,params=None,energ=None,fold=True):    
        """
        This method is used to evaluate and return the model values for a given 
        set of parameters,  over a given model energy grid. By default it  
        will evaluate the model over the energy grid defined in the response,
        using the parameters values stored internally in the model_params 
        attribute, without folding the model through the response.        
        
        Parameters:
        -----------                         
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are 
            provided, the model_params attribute is used.
            
        energ: np.array(float), default None
            The the photon energies over which to evaluted the model. If 
            none are provided, the same grid contained in the instrument response  
            is used. 
            
        fold: bool, default True
            A boolean switch to choose whether to fold the evaluated model 
            through the instrument response or not. Not that in order for the 
            model to be folded, the energy grid over which it is defined MUST 
            be identical to that stored in the response matrix/class.
            
        Returns:
        --------
        model: np.array(float)
            The model evaluated over the given energy grid, for the given input 
            parameters.  
        """    
    
        if energ is None:
            energ = self.energs
        if params is None:
            model = self.model.eval(self.model_params,energ=energ)*self.energ_bounds
        else:
            model = self.model.eval(params,energ=energ)*self.energ_bounds
        if fold is True:
            model = self.response.convolve_response(model)  
        return model

    def get_residuals(self,model,res_type):    
        """
        This methods return the residuals (either as data/model, or as 
        contribution to the total chi squared) of the input model, given the 
        parameters set in model_parameters, with respect to the data. 
        
        Parameters:
        -----------
        model_vals: np.array(float)
            An array of model values to be compared against the data.
            
        res_type: string 
            If set to "ratio", the method returns the residuals defined as 
            data/model. If set to "delchi", it returns the contribution of 
            each energy channel to the total chi squared.
            
        Returns:
        --------
        residuals: np.array(float)
            An array of the same size as the data, containing the model 
            residuals in each channel.
            
        bars: np.array(float)
            An array of the same size as the residuals, containing the one sigma 
            range for each contribution to the residuals.           
        """
        
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
        """
        This method compares the model defined by the user, using the last set
        of parameters to have been set in the class, to the data stored. It then
        prints the chi-squared goodness-of-fit to terminal, along with the 
        number of data bins, free parameters and degrees of freedom. 
        """
        
        if self.likelihood is None:
            res, err = self.get_residuals(model,"delchi")
            chi_squared = np.sum(np.power(res.reshape(len(self.data)),2))
            freepars = 0
            for key, value in self.model_params.items():
                param = self.model_params[key]
                if param.vary is True:
                    freepars += 1
            dof = len(self.data) - freepars
            reduced_chisquared = chi_squared/dof
            print("Goodness of fit metrics:")
            print("Chi squared" + "{0: <13}".format(" ") + str(chi_squared))
            print("Reduced chi squared" + "{0: <5}".format(" ") + str(reduced_chisquared))
            print("Data bins:" + "{0: <14}".format(" ") + str(len(self.data)))
            print("Free parameters:" + "{0: <8}".format(" ") + str(freepars))
            print("Degrees of freedom:" + "{0: <5}".format(" ") + str(dof))
        else:
            print("custom likelihood not supported yet")
        return 

    def _spectrum_minimizer(self,params):
        """
        This method is used exclusively when running a minimization algorithm.
        It evaluates the model for an input set of parameters, and then returns 
        the residuals in units of contribution to the total chi squared 
        statistic.
        
        Parameters:
        -----------                         
        params: lmfit.Parameters
            The parameter values to use in evaluating the model. These will vary 
            as the fit runs.
            
        Returns:
        --------
        residuals: np.array(float)
            An array of the same size as the data, containing the model 
            residuals in each bin.            
        """
    
        if self.likelihood is None:
            model = self.eval_model(params,energ=self.energs)
            convolve = np.extract(self.ebounds_mask,model)
            residuals = (self.data-convolve)/self.data_err
        else:
            raise TypeError("custom likelihood not implemented yet")
        return residuals

    def fit_data(self,algorithm='leastsq'):
        """
        This method attempts to minimize the residuals of the model with respect 
        to the data defined by the user. The fit always starts from the set of 
        parameters defined with .set_params(). Once the algorithm has completed 
        its run, it prints to terminal the best-fitting parameters, fit 
        statistics, and simple selection criteria (reduced chi-squared, Akaike
        information criterion, and Bayesian informatino criterion). 
        
        Parameters:
        -----------
        algorithm: str, default="leastsq"
            The fitting algorithm to be used in the minimization. The possible 
            choices are detailed on the LMFit documentation page:
            https://lmfit.github.io/lmfit-py/fitting.html#fit-methods-table.
        """
        
        self.fit_result = minimize(self._spectrum_minimizer,self.model_params,
                                   method=algorithm)
        print(fit_report(self.fit_result,show_correl=False))
        fit_params = self.fit_result.params
        self.set_params(fit_params)
        return

    def plot_data(self,units="data",return_plot=False):
        """
        This method plots the spectrum loaded by the user as a function of 
        energy. It is possible to plot both in detector and ``unfolded'' space, 
        with the caveat that unfolding data is EXTREMELY dangerous and should
        be interpreted with care (or not at all). 
        
        The definition of unfolded data is subjective; nDspec adopts the same 
        convention as ISIS, and defines an unfolded count spectrum Uf(h) as a 
        function of energy channel h as :
        Uf(h) = C(h)/sum(R(E)),
        where C(h) is the detector space spectrum, R(E) is the instrument response 
        and sum denotes the sum over energy bins. This definition has the 
        advantage of being model-independent and is analogous to the Xspec 
        (model-dependent) definition of unfolding data when the model is a 
        constant. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        units: str, default="data"
            The units to use for the y axis. units="data", the detector, plots 
            the data in detector space in units of counts/s/keV. units="unfold" 
            instead plots unfolded data and follows the Xspec convention for the 
            y axis - the y axis is in units of counts/s/keV/cm^2, times one 
            additional factor "keV" for each "e" that appears in the string. 
            For instance, units="eeunfold" plots units of kev^2 counts/s/keV/cm^2,
            i.e. units of nuFnu. 
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
    
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
        """
        This method plots the model defined by the user as a function of 
        energy, as well as (optionally) its components, and the data plus model
        residuals. It is possible to plot both in detector and ``unfolded'' space, 
        with the caveat that unfolding data is EXTREMELY dangerous and should
        be interpreted with care (or not at all). 
        
        The definition of unfolded data is subjective; nDspec adopts the same 
        convention as ISIS, and defines an unfolded count spectrum Uf(h) as a 
        function of energy channel h as :
        Uf(h) = C(h)/sum(R(E)),
        where C(h) is the detector space spectrum, R(E) is the instrument response 
        and sum denotes the sum over energy bins. This definition has the 
        advantage of being model-independent and is analogous to the Xspec 
        (model-dependent) definition of unfolding data when the model is a 
        constant. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        plot_data: bool, default=True
            If true, both model and data are plotted; if false, just the model. 
            
        plot_components: bool, default=False 
            If true, the model components are overplotted; if false, they are 
            not. Only additive model components will display their values 
            correctly. 
            
        params: lmfit.parameters, default=None 
            The parameters to be used to evaluate the model. If False, the set 
            of parameters stored in the class is used 
        
        units: str, default="data"
            The units to use for the y axis. units="data", the detector, plots 
            the data in detector space in units of counts/s/keV. units="unfold" 
            instead plots unfolded data and follows the Xspec convention for the 
            y axis - the y axis is in units of counts/s/keV/cm^2, times one 
            additional factor "keV" for each "e" that appears in the string. 
            For instance, units="eeunfold" plots units of kev^2 counts/s/keV/cm^2,
            i.e. units of nuFnu. 
            
        residuals: str, default="delchi"
            The units to use for the residuals. If residuals="delchi", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """           
                                     
        energies = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        xerror = 0.5*np.extract(self.ebounds_mask,self._ewidths_unmasked)       
        
        #first; get the model in the correct units
        #need to be careful about the units here - although all this mess is solved by 
        #going to the xspec convention for energy dependence...
        model_fold = self.eval_model(params=params,energ=self.energs)
        #model_fold = self.response.convolve_response(model_prefold)
        if units == "data":   
            model = np.extract(self.ebounds_mask,model_fold)   
            ylabel = "Folded counts/s/keV"
        elif units.count("unfold"):
            power = units.count("e")
            #the reason for folding and then unfolding the model is that 
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
                data = self.response.unfold_response(self._data_unmasked)* \
                       self._ebounds_unmasked**power
                error = self.response.unfold_response(self._data_err_unmasked)* \
                        self._ebounds_unmasked**power  
                data = np.extract(self.ebounds_mask,data)
                yerror = np.extract(self.ebounds_mask,error)
            
        if plot_data is False:
            fig, (ax1) = plt.subplots(1,1,figsize=(6.,4.5))   
        else:
            fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6.,6.),
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
                comp_folded = self.response.convolve_response(comps[key]*self.energ_bounds)
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
            ax1.set_ylim([0.85*np.min(data),1.15*np.max(data)])
            ax2.errorbar(energies,model_res,yerr=res_errors,
                         drawstyle="steps-mid",marker='o')
            if residuals == "delchi":
                ax2.plot(energies,np.zeros(len(energies)),
                         ls=":",lw=2,color='black')
            elif residuals == "ratio":
                ax2.plot(energies,np.ones(len(energies)),
                         ls=":",lw=2,color='black')                
            ax2.set_xlabel("Energy (keV)")
            ax2.set_ylabel(reslabel)

        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return  

class Fit_OneDCrossSpectrum():
    """
    Least-chi squared fitter class for a one dimensional cross spectrum between
    two energy bands, defined as the product between the Fourier transform of 
    a subjet band lightcurve, and the complex conjugate of the transform of a 
    reference lightcurve. 
    
    Given an array of Fourier frequencies, a cross spectrum, its error and a 
    model (defined in Fourier space), this class handles fitting internally 
    using the lmfit library. The model can be defined for the cross spectrum 
    expicitely, or it can be an impulse response (or transfer) function which 
    can then be converted into a cross spectrum. In the latter case, users also 
    need to specify a form for the power spectrum.    
    
    Poisson noise in the data is not accounted for explicitely. Users should 
    either pass noise-subtracted data, or include a constant component in the 
    model definition (see below).   
        
    Attributes
    ----------
    model: lmfit.CompositeModel 
        A lmfit CompositeModel object, which contains a wrapper to the model 
        component(s) one wants to fit to the data. 
        
    model_type: str,
   
    model_params: lmfit.Parameters 
        A lmfit Parameters object, which contains the parameters for the model 
        components.
   
    likelihood: None
        Work in progress; currently the software defaults to chi squared 
        likelihood
   
    fit_result: lmfit.MinimizeResult
        A lmfit MinimizeResult, which stores the result (including best-fitting 
        parameter values, fit statistics etc) of a fit after it has been run.         
   
    data: np.array(float)
        An array of float containing the cross spectrum to be fitted. Users can 
        pass data in polar coordinates (modulus and phase), cartesian coordinates
        (real and imaginary), or purely time lags (lags). 
   
    data_err: np.array(float)
        The uncertainty on the cross spectrum stored in data. 
   
    freqs: np.array(float)
        The Fourier frequency over which both the data and model are defined, 
        in units of Hz.   
        
    response: nDspec.ResponseMatrix
        The instrument response matrix corresponding to the spectrum to be 
        fitted. It is required to define the energy grids over which model and
        data are defined. 
   
    energs: np.array(float)
        The array of physical photon energies over which the model is computed. 
        Defined as the middle of each bin in the energy range stored in the 
        instrument response provided.   
        
    ref_band: list[float,float]
        The energy range for the reference band, in keV.
        
    sub_band: list[float,float]
        The energy range for the subject band, in keV. 
        
    _times: np.array(float)
        The array of time stamps corresponding to the Fourier frequency grid. 
        Used exclusively for internal book keeping. 
        
    powerspec: nDspec.PowerSpectrum
        A nDspec PowerSpectrum object used to calculate the model cross spectrum 
        from an impulse response or transfer function. 
        
    crossspec: nDspec.CrossSpectrum
        A nDspec CrossSpectrum object used to store the model cross-spectrum 
        values and convert to whatever units are appropriate (e.g. lags, 
        phase/modulus, etc). 
        
    units: str
        A string specifying whether the units of both the data and model are 
        cartesian (real and imaginary parts), polar (modulus and phase), or 
        just Fourier lags (converted from the phase). 
        
    _supported_units, _supported_models: str
        A list of strings specifying the data and model units, or the type of
        model. Required for internal book keeping. 
    """ 

    def __init__(self):
        self.model = None
        self.model_type = None
        self.model_params = None
        self.likelihood = None
        self.fit_result = None
        self.data = None
        self.data_err = None
        self.response = None
        self.energs = None
        self.ref_band = None
        self.sub_band = None
        #self.emin = None
        #self.emax = None
        #self.ebounds = None
        #self.ewidths = None
        self.freqs = None 
        self._times = None
        self.powerspec = None
        self.crossspec = None
        self.units = None
        self._supported_units = ["cartesian","polar","lags"]
        self._supported_models = ["cross","irf","transfer"]
        pass

    #tbd: allow people to ignore frequency ranges
    def set_units(self,units="cartesian"):
        """
        This method is used to specify which units will be used for the data and 
        model. The supported units are cartesian (real and imaginary parts), polar 
        (modulus and phase), or lags (which take the phase and convert to a time).
        
        Parameters:
        -----------                         
        units: str, default="cartesian"
            The units to be used for data and model. Supported units are 
            cartesian, polar and lags.                
        """
    
        if units not in self._supported_units:
            raise AttributeError("Unsopprted units for the cross spectrum")
        else:
            self.units = units
        return 
    
    def set_data(self,response,ref_band,sub_band,time_res,seg_size,norm,
                 data,data_err=None,data_grid=None):
        """
        This method is used to pass the data users want to fit. The user needs 
        to specify an instrument response, reference and subject bands, the time 
        resolution and lightcurve segment size used to build the data, as well 
        as the desired normalization (if the cross spectrum is built with stingray, 
        see below). 
        
        The data can be passed as a stingray.events object, in which case the
        method will calculate the errors and renormalize the data automatically.
        Alternatively, users can explicitely pass arrays with the cross spectrum,
        its error, and the Fourier frequency grid. In this case, the data is 
        assumed to already have been re-normalized to the users' preferred units.
        
        Parameters:
        -----------
        response: nDspec.ResponseMatrix
            The instrument response matrix corresponding to the spectrum to be 
            fitted. It is required to define the energy grids over which model and
            data are defined. 
            
        ref_band: list[float,float]
            The energy range for the reference band, in keV.
            
        sub_band: list[float,float]
            The energy range for the subject band, in keV. 
            
        time_res: float 
            The time resolution of the lightcurves used to build the cross 
            spectrum.
            
        seg_size: float 
            The lightcurve segment duration used to build the cross spectrum.
            
        norm: str
            The normalization of the cross spectrum built from the provided 
            stingray event file. Can be either "abs" for absolute-rms normalization,
            or "frac" for fractional rms normalization. 
                   
        data: np.array(float) or stingray.powerspectrum
            The power spectrum to be fitted, either as a numpy array or a 
            stingray event file object.
            
        data_err: np.array(float), default: None 
            The error on the cross spectrum. If passing a stingray object, this 
            is not necessary and is therefore ignored.
            
        data_grid: np.array(float), default: None 
            The Fourier frequency grid over which the data (and model) are 
            defined. If passing a stingray object, this is not necessary and is 
            therefore ignored.      
        """
        
        #will need to rebin this to only take the bounds in the reference and channel of interest, but it's fine for now 
        self.response = response#.rebin_channels(bounds_lo,bounds_hi)
        self.energs = 0.5*(self.response.energ_hi+self.response.energ_lo)
        self.ewidths = self.response.energ_hi-self.response.energ_lo
        self.ref_band = ref_band
        self.sub_band = sub_band
        #I might need the energy bins too
        if self.units is None:
            raise AttributeError("Cross spectrum units not defined")
        if data.__module__ == "stingray.events":
            events_ref = events.filter_energy_range(ref_band)
            events_sub = events.filter_energy_range(sub_band)
            #get the cross spectrum
            cs = AveragedCrossspectrum.from_events(events_sub,events_ref,
                                                   segment_size=seg_size,dt=time_res,norm=norm)
            self.freqs = cs.freq
            #set the time axis from the Fourier frequency axis
            #this is needed for the ndspec objects
            lc_length = cs.n*time_res
            time_samples = int(lc_length/time_res)
            self._times = np.linspace(time_res,lc_length,time_samples)
            if self.units == "lags":
                self.data, self.data_err = cs.time_lag()   
            else:
                #get all the things needed for the error bars
                ps_sub = AveragedPowerspectrum.from_events(events_sub,
                                                           segment_size=seg_size,
                                                           dt=time_res,norm=norm)
                ps_ref = AveragedPowerspectrum.from_events(events_ref,
                                                           segment_size=seg_size,
                                                           dt=time_res,norm=norm)
                ctrate_sub = get_average_ctrate(events_sub.time,events_sub.gti,seg_size)
                ctrate_ref = get_average_ctrate(events_ref.time,events_ref.gti,seg_size)
                noise_sub = poisson_level(norm=norm, meanrate=ctrate_sub)
                noise_ref = poisson_level(norm=norm, meanrate=ctrate_ref)
                if norm == "frac":
                    N = 2./noise_ref 
                elif norm == "abs":
                    N = 2.*noise_sub
                else:
                    print("Normalization is wrong")
                    N = 1.
            if self.units == "cartesian":    
                data_real = np.real(cs.power)
                data_imag = np.imag(cs.power)                
                error_real = np.sqrt((ps_sub.power*ps_ref.power+ \
                                      np.real(cs.power)**2- \
                                      np.imag(cs.power)**2)/(2.*N))
                error_imag = np.sqrt((ps_sub.power*ps_ref.power- \
                                      np.real(cs.power)**2+ \
                                      np.imag(cs.power)**2)/(2.*N))
                self.data = np.concatenate((data_real,data_imag))
                self.data_err = np.concatenate((error_real,error_imag))
            elif self.units == "polar":
                data_mod = np.absolute(cs.power)
                error_mod = np.sqrt(ps_sub.power*ps_ref.power/(2.*N))
                data_phase, error_phase = cs.phase_lag()
                self.data = np.concatenate((data_mod,data_phase))
                self.data_err = np.concatenate((error_mod,error_phase))
        else:
            self.data = data
            self.data_err = data_err
            self.freqs = data_grid
            #now revert the grid from frequency to time, and save in times
            #this is needed to allocate the ndspec objects
            time_res = 0.5/(self.freqs[-1]+self.freqs[0])
            lc_length = (self.freqs.size+1)*2*time_res
            time_samples = int(lc_length/time_res)
            self._times = np.linspace(time_res,lc_length,time_samples)
        return 

    def set_model(self,model,model_type="irf",params=None):
                """
        This method is used to pass the model users want to fit to the data. 
        Optionally it is also possible to pass the initial parameter values of 
        the model and the type of model.
        
        Three types of models are currently supported. "cross" is used for models
        expclitely for the cross spectrum. "irf" is used for models of impulse 
        response functions, defined in the time domain. "transfer" is used for 
        models of transfer functions, defined as the Fourier transform of an 
        impulse response function. In the latter two cases, users also need to 
        supply an input power spectrum with the set_psd_weights method. 
        
        Parameters:
        -----------            
        model: lmfit.CompositeModel 
            The lmfit wrapper of the model one wants to fit to the data. 
            
        model_type: str, default: "irf"
            The type of model defined for the data. "cross" applies directly 
            to the cross spectrum, "irf" defines a time-domain impulse response 
            function, "transfer" defines a Fourier-domain transfer function.           
            
        params: lmfit.Parameters, default: None 
            The parameter values from which to start evalauting the model during
            the fit. If it is not provided, all model parameters will default 
            to 0, set to be free, and have no minimum or maximum bound. 
        """
        
        if model_type not in self._supported_models:
            raise AttributeError("Unsopprted model type")  
        self.model_type = model_type
        self.crossspec = CrossSpectrum(self._times,freqs=self.freqs,energ=self.energs)
        if self.model_type != "cross":
            self.powerspec = PowerSpectrum(self._times)        
        self.model = model 
        if params is None:
            self.model_params = self.model.make_params(verbose=True)
        else:
            self.model_params = params
        return 
        
    def set_psd_weights(self,psd_weights):
        """  
        This method sets the weighing power spectrum used to convert a model 
        impulse response or transfer function into a cross spectrum. 
        
        Parameters
        ----------
        input_power: np.array(float) or PowerSpectrum
            Either an array of size (n_freqs) that is to be used as the weighing  
            power spectrum when computing the cross spectrum, or an nDspec 
            PowerSpectrum object. Both have to be defined over the same Fourier 
            frequency array as the data. 
        """
        
        if self.model_type != "cross":
            self.crossspec.set_psd_weights(psd_weights)
        else:
            print("Power spectrum weight not needed")
        return 

    def set_params(self,params):
        """
        This method is used to set the model parameter names and values. It can
        be used both to initialize a fit, and to test different parameter values 
        before actually running the minimization algorithm.
        
        Parameters:
        -----------                       
        params: lmfit.Parameters
            The parameter values from which to start evalauting the model during
            the fit.  
        """
        self.model_params = params
        return 
    
    def eval_model(self,params=None,ref_band=None,sub_band=None):
        """
        This method is used to evaluate and return the model values for a given 
        set of parameters, over the input energy/frequency grids, for a user-provided
        combination of reference and subject bands. By default it use the reference 
        and subject bands defined when loading the data, using the parameters 
        values stored internally in the model_params attribute.       
        
        Parameters:
        -----------                         
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are 
            provided, the model_params attribute is used.
            
        ref_band: list[float,float]
            The energy range for the reference band, in keV.
            
        sub_band: list[float,float]
            The energy range for the subject band, in keV. 
            
        Returns:
        --------
        model: np.array(float)
            The model evaluated for the given input parameters. If the units are 
            cartesian or polar, the size of the array is twice that of the Fourier 
            frequency array, and it contains (real, imaginary) or (modulus, phase) 
            parts of the cross spectrum, respectively. If the units are lags, 
            the size of the array is identical to the Fourier frequency grid, 
            and each bin contains the time lag in that bin.  
        """        
        
        if ref_band is None:
            ref_band = self.ref_band
        if sub_band is None:
            sub_band = self.sub_band
        
        #evaluate the model for the chosen parameters
        if params is None:
            params= self.model_params
        model_eval = self.model.eval(params,freqs=self.freqs,energs=self.energs)#*self.energ_bounds

        #store the model in the cross spectrum, depending on the type
        if self.model_type == "irf":
            self.crossspec.set_impulse(np.transpose(model_eval))
            self.crossspec.set_reference_energ(ref_band)
            self.crossspec.cross_from_irf()
        elif self.model_type == "transfer":
            self.crossspec.set_transfer(np.transpose(model_eval))
            self.crossspec.set_reference_energ(ref_band)
            self.crossspec.cross_from_transfer()
        else:
            #transposing is required to ensure the units are correct 
            self.crossspec.cross = np.transpose(model_eval)
        
        #fold the instrument response:
        folded_eval = self.response.convolve_response(self.crossspec,units_in="rate",units_out="kev")

        #depending on units, return the correct format
        if self.units == "lags":
            model = folded_eval.lag_frequency(self.sub_band)
        elif self.units == "cartesian":
            real = folded_eval.real_frequency(self.sub_band)
            imag = folded_eval.imag_frequency(self.sub_band)
            model = np.concatenate((real,imag))
        elif self.units == "polar":
            mod = folded_eval.mod_frequency(self.sub_band)
            phase = folded_eval.phase_frequency(self.sub_band)
            model = np.concatenate((mod,phase))
        else:
            print("weird units raise proper error tbd")
        return model

    def get_residuals(self,model,res_type):
        """
        This methods return the residuals (either as data/model, or as 
        contribution to the total chi squared) of the input model, given the 
        parameters set in model_parameters, with respect to the data. 
        
        Parameters:
        -----------
        model_vals: np.array(float)
            An array of model values to be compared against the data.
            
        res_type: string 
            If set to "ratio", the method returns the residuals defined as 
            data/model. If set to "delchi", it returns the contribution of 
            each energy channel to the total chi squared.
            
        Returns:
        --------
        residuals: np.array(float)
            An array of the same size as the data, containing the model 
            residuals in each channel.
            
        bars: np.array(float)
            An array of the same size as the residuals, containing the one sigma 
            range for each contribution to the residuals.           
        """
        
        model = self.eval_model()
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
        """
        This method compares the model defined by the user, using the last set
        of parameters to have been set in the class, to the data stored. It then
        prints the chi-squared goodness-of-fit to terminal, along with the 
        number of data bins, free parameters and degrees of freedom. 
        """
    
        if self.likelihood is None:
            res, err = self.get_residuals(model,"delchi")
            chi_squared = np.sum(np.power(res.reshape(len(self.data)),2))
            freepars = 0
            for key, value in self.model_params.items():
                param = self.model_params[key]
                if param.vary is True:
                    freepars += 1
            dof = len(self.data) - freepars
            reduced_chisquared = chi_squared/dof
            print("Goodness of fit metrics:")
            print("Chi squared" + "{0: <13}".format(" ") + str(chi_squared))
            print("Reduced chi squared" + "{0: <5}".format(" ") + str(reduced_chisquared))
            print("Data bins:" + "{0: <14}".format(" ") + str(len(self.data)))
            print("Free parameters:" + "{0: <8}".format(" ") + str(freepars))
            print("Degrees of freedom:" + "{0: <5}".format(" ") + str(dof))
        else:
            print("custom likelihood not supported yet")
        return 
    
    def _cross_minimizer(self,params):
        """
        This method is used exclusively when running a minimization algorithm.
        It evaluates the model for an input set of parameters, and then returns 
        the residuals in units of contribution to the total chi squared 
        statistic.
        
        Parameters:
        -----------                         
        params: lmfit.Parameters
            The parameter values to use in evaluating the model. These will vary 
            as the fit runs.
            
        Returns:
        --------
        residuals: np.array(float)
            An array containing the model residuals in each frequency bin. If 
            the units are  cartesian or polar, the size of the array is twice
            that of the Fourier  frequency array, and it contains (real, imaginary) 
            or (modulus, phase) parts of the cross spectrum, respectively. If 
            the units are lags,  the size of the array is identical to the 
            Fourier frequency grid,  and each bin contains the time lag in that 
            bin.             
        """
    
        if self.likelihood is None:
            model = self.eval_model(params,ref_band=self.ref_band,sub_band=self.sub_band)
            residuals = (self.data-model)/self.data_err
        else:
            raise TypeError("custom likelihood not implemented yet")
        return residuals
    
    def fit_data(self,algorithm='leastsq'):
        """
        This method attempts to minimize the residuals of the model with respect 
        to the data defined by the user. The fit always starts from the set of 
        parameters defined with .set_params(). Once the algorithm has completed 
        its run, it prints to terminal the best-fitting parameters, fit 
        statistics, and simple selection criteria (reduced chi-squared, Akaike
        information criterion, and Bayesian informatino criterion). 
        
        Parameters:
        -----------
        algorithm: str, default="leastsq"
            The fitting algorithm to be used in the minimization. The possible 
            choices are detailed on the LMFit documentation page:
            https://lmfit.github.io/lmfit-py/fitting.html#fit-methods-table.
        """
    
        self.fit_result = minimize(self._cross_minimizer,self.model_params,
                                   method=algorithm)
        print(fit_report(self.fit_result,show_correl=False))
        fit_params = self.fit_result.params
        self.set_params(fit_params)
        return    
    
    def plot_data(self,return_plot=False):
        """
        This method plots the cross spectrum loaded by the user as a function of 
        Fourier frequency. The units used to plot the data are the same as those
        used to define the data. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------           
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
    
        data_bins = len(self.freqs)

        if self.units != "lags":
            fig, ((ax1),(ax2)) = plt.subplots(1,2,figsize=(12.,5.))  
            
            ax1.errorbar(self.freqs,self.data[:data_bins]*self.freqs,
                         yerr=self.data_err[:data_bins]*self.freqs,
                         drawstyle="steps-mid",
                         marker='o',
                         zorder=2,
                         color='C0')
            ax1.set_xscale("log")
            ax1.set_xlabel("Frequency (Hz)")
            if self.units == "cartesian":
                ylabel = "Real part$\\times$Freq"
                ax1.axhline(0, ls="dotted",color='black')
            else:
                ylabel = "Modulus$\\times$Freq"
                ax1.set_yscale("log")
            ax1.set_ylabel(ylabel)
            if self.units == "cartesian":
                ylabel = "Imaginary part$\\times$Freq"
                ax2.errorbar(self.freqs,self.data[data_bins:]*self.freqs,
                             yerr=self.data_err[data_bins:]*self.freqs,
                             drawstyle="steps-mid",
                             marker='o',
                             zorder=2,
                             color='C0')
            else:
                ylabel = "Phase"
                ax2.errorbar(self.freqs,self.data[data_bins:],
                             yerr=self.data_err[data_bins:],
                             drawstyle="steps-mid",
                             marker='o',
                             zorder=2,
                             color='C0')
            ax2.axhline(0, ls="dotted",color='black')
            ax2.set_xscale("log")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel(ylabel)
        else:
            ylabel = "Lag (s)"
            fig, ((ax1)) = plt.subplots(1,1,figsize=(6.,4.5))         
            ax1.errorbar(self.freqs,self.data,
                         yerr=self.data_err,
                         drawstyle="steps-mid",
                         marker='o',
                         zorder=2,
                         color='C0')
            ax1.axhline(0, ls="dotted",color='black')
            ax1.set_ylabel(ylabel)
            ax1.set_xlabel("Frequency (Hz)")         
            ax1.set_xscale("log")
        
        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return 

    def plot_model(self,plot_data=True,params=None,residuals="delchi",return_plot=False):
        """
        This method plots the model defined by the user as a function of 
        Fourier frequency, as well as (optionally) its components, and the data
        plus model residuals. The units used to plot the data are the same as 
        those used to define the data.
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        plot_data: bool, default=True
            If true, both model and data are plotted; if false, just the model. 
            
        plot_components: bool, default=False 
            If true, the model components are overplotted; if false, they are 
            not. Only additive model components will display their values 
            correctly. 
            
        params: lmfit.parameters, default=None 
            The parameters to be used to evaluate the model. If False, the set 
            of parameters stored in the class is used 
            
        residuals: str, default="delchi"
            The units to use for the residuals. If residuals="delchi", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
    
        data_bins = len(self.freqs) 
        model = self.eval_model(params=params)

        #if we're plotting data, also get the residuals
        if plot_data is True:
            model_res,res_errors = self.get_residuals(self.model,residuals)
            if residuals == "delchi":
                reslabel = "$\\Delta\\chi$"
            else:
                reslabel = "Data/model"     

        if self.units != "lags":
            if plot_data is True:
                fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12.,6.),
                                                          sharex=True,
                                                          gridspec_kw={'height_ratios': [2, 1]})  
                ax1.errorbar(self.freqs,self.data[:data_bins]*self.freqs,
                             yerr=self.data_err[:data_bins]*self.freqs,
                             drawstyle="steps-mid",
                             marker='o',
                             zorder=2) 
                ax3.errorbar(self.freqs,model_res[:data_bins],
                             yerr=res_errors[:data_bins],
                             drawstyle="steps-mid",
                             marker='o',
                             zorder=2)
                ax3.set_xlabel("Frequency (Hz)") 
                ax3.set_ylabel(reslabel)
                ax4.errorbar(self.freqs,model_res[data_bins:],
                             yerr=res_errors[data_bins:],
                             drawstyle="steps-mid",
                             marker='o',
                             zorder=2)
                ax4.set_ylabel(reslabel)
                ax4.set_xlabel("Frequency (Hz)") 
                if residuals == "delchi":
                    ax3.plot(self.freqs,np.zeros(len(self.freqs)),ls=":",lw=2,color='black')
                    ax4.plot(self.freqs,np.zeros(len(self.freqs)),ls=":",lw=2,color='black')
                elif residuals == "ratio":
                    ax3.plot(self.freqs,np.zeros(len(self.freqs)),ls=":",lw=2,color='black')                    
                    ax4.plot(self.freqs,np.ones(len(self.freqs)),ls=":",lw=2,color='black')                   
                if self.units == "polar":
                    ax2.errorbar(self.freqs,self.data[data_bins:],
                                 yerr=self.data_err[data_bins:],
                                 drawstyle="steps-mid",
                                 marker='o',
                                 zorder=2)  
                else:
                    ax2.errorbar(self.freqs,self.data[data_bins:]*self.freqs,
                                 yerr=self.data_err[data_bins:]*self.freqs,
                                 drawstyle="steps-mid",
                                 marker='o',
                                 zorder=2)                     
            else:
                fig, ((ax1),(ax2)) = plt.subplots(1,2,figsize=(12.,5.))  
                ax1.set_xlabel("Frequency (Hz)") 
                ax2.set_xlabel("Frequency (Hz)")  
            ax1.plot(self.freqs,model[:data_bins]*self.freqs,linewidth=3,zorder=3)
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            if self.units == "polar":
                ax1.set_ylabel("Modulus$\\times$Freq")
                ax2.set_ylabel("Phase")
                ax2.plot(self.freqs,model[data_bins:],linewidth=3,zorder=3)
            else:
                ax1.set_ylabel("Real part$\\times$Freq")
                ax2.set_ylabel("Imaginary part$\\times$Freq")
                ax2.plot(self.freqs,model[data_bins:]*self.freqs,linewidth=3,zorder=3)
            ax2.plot(self.freqs,np.zeros(len(self.freqs)),ls=":",lw=2,color='black')
            ax2.set_xscale("log")
        else:
            #if we're plotting the data, automatically also plot the residuals
            if plot_data is True:
                fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6.,6.),
                              sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})
                ax1.errorbar(self.freqs,self.data,
                             yerr=self.data_err,
                             drawstyle="steps-mid",
                             marker='o',
                             zorder=2)    
                ax2.errorbar(self.freqs,model_res,
                             yerr=res_errors,
                             drawstyle="steps-mid",
                             marker='o',
                             zorder=2)  
                ax2.set_xlabel("Frequency (Hz)")  
                if residuals == "delchi":
                    ax2.plot(self.freqs,np.zeros(len(self.freqs)),ls=":",lw=2,color='black')
                elif residuals == "ratio":
                    ax2.plot(self.freqs,np.ones(len(self.freqs)),ls=":",lw=2,color='black')   
                ax2.set_ylabel(reslabel)
            else:
                fig, (ax1) = plt.subplots(1,1,figsize=(6.,4.5),sharex=True,
                                          gridspec_kw={'height_ratios': [2, 1]})
                ax1.set_xlabel("Frequency (Hz)")  
            ylabel = "Lag (s)" 
            ax1.errorbar(self.freqs,model,linewidth=3,zorder=3)
            ax1.axhline(0, ls="dotted",color='black')
            ax1.set_ylabel(ylabel)
            ax1.set_xscale("log")

        plt.tight_layout()    
        
        if return_plot is True:
            return fig 
        else:
            return  
        
def load_pha(path,response):
    '''
    This function loads an X-ray spectrum , given an input path to an OGIP-compatible
    file and a nDspec ResponseMatrix object to be applied to the spectrum. 
  
    
    Parameters:
    -----------
    path: str 
        A string pointing to the spectrum file to be loaded 
        
    response: nDspec.ResponseMatrix 
        The instrument response matrix, loaded in nDspec, corresponding to the 
        spectrum to be loaded 
        
    Returns:
    --------
    bin_bounds_lo: np.array(float)
        An array of lower energy channel bounds, in keV, as contained in the 
        input file. If the spectrum was grouped, this contains the lower bounds 
        of the spectrum after rebinning.
        
    bin_bounds_hi: np.array(float)
        An array of upper energy channel bounds, in keV, as contained in the 
        input file. If the spectrum was grouped, this contains the lower bounds 
        of the spectrum after rebinning.
        
    counts_per_group: np.array(float)
        The total number of photon counts in each energy channel. If the spectrum 
        was grouped, this contains the counts in each channel after rebinning. 
        
    spectrum_error: np.array(float)
        The error on the counts in each group, including both Poisson and (if
        present) systematic errors
        
    exposure: float
        The exposure time contained in the spectrum file.        
    '''
    from astropy.io import fits
    
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

def loadr_lc(path):
    '''
    This function loads an X-ray lightcurve, given an input path to an 
    OGIP-compatible file.
    
    Parameters:
    -----------
    path: str 
        A string pointing to the lightcurve file to be loaded 
   
    Returns:
    --------
    time_bins: np.array(float)
        An array of time stamps covered by the lightcurve 
        
    counts: np.array(float) 
        An array of counts rates (defined in counts per second) contained in the 
        lightcurve 
        
    gti: list([float,float])
        A list of good time intervals over which the lightcurve is defined. 
    '''
    from astropy.io import fits

    with fits.open(path,filemap=False) as lightcurve:
        extnames = np.array([h.name for h in lightcurve])
        lightcurve_data = lightcurve['RATE'].data
        time_bins = lightcurve_data['TIME']
        counts = lightcurve_data['RATE']
        gti_data = lightcurve['GTI'].data
        gti = [gti_data['START']-gti_data['START'][0],gti_data['STOP']-gti_data['START'][0]]

    return time_bins, counts, gti
