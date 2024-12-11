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

class SimpleFit():
    """
    Generic least-chi squared fitter class, used internally to store methods 
    that are shared between all the fitter types. 
           
    Attributes:
    -----------
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
        An array storing the data to be fitted. If the data is complex and/or 
        multi-dimensional, it is flattened to a single dimension in order to be 
        compatible with the LMFit fitter methods.
   
    data_err: np.array(float)
        An array containing the uncertainty on the data to be fitted. It is also 
        stored as a one-dimensional array regardless of the type or dimensionality 
        of the initial data.           
    """ 

    def __init__(self):
        self.model = None
        self.model_params = None
        self.likelihood = None
        self.fit_result = None
        self.data = None
        self.data_err = None
    pass

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

    def get_residuals(self,model,res_type,use_masked=True):    
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

        if (isinstance(model,LM_Model)):
            model = self.eval_model()
        
        if use_masked is True:
            data = self.data
            error = self.data_err
        elif use_masked is False:
            data = self._data_unmasked
            error = self._data_err_unmasked
        
        if isinstance(self,FitTimeAvgSpectrum):
            model = np.extract(self.ebounds_mask,model)
        if res_type == "ratio":
            residuals = data/model
            bars = error/model
        elif res_type == "delchi":
            residuals = (data-model)/error
            bars = np.ones(len(data))
        else:
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

    def fit_data(self,algorithm='leastsq'):
        """
        This method attempts to minimize the residuals of the model with respect 
        to the data defined by the user. The fit always starts from the set of 
        parameters defined with .set_params(). Once the algorithm has completed 
        its run, it prints to terminal the best-fitting parameters, fit 
        statistics, and simple selection criteria (reduced chi-squared, Akaike
        information criterion, and Bayesian information criterion). 
        
        Parameters:
        -----------
        algorithm: str, default="leastsq"
            The fitting algorithm to be used in the minimization. The possible 
            choices are detailed on the LMFit documentation page:
            https://lmfit.github.io/lmfit-py/fitting.html#fit-methods-table.
        """
        
        self.fit_result = minimize(self._minimizer,self.model_params,
                                   method=algorithm)
        print(fit_report(self.fit_result,show_correl=False))
        fit_params = self.fit_result.params
        self.set_params(fit_params)
        return

class EnergyDependentFit():
    """
    Internal book-keeping class used to manage noticing or ignoring energy 
    channels, for cases when the data requires an instrument response. 
    
    Stores the full (unmasked) energy center/bounds, and data arrays, a mask
    used to track which channels/data points are noticed or ignored, as well as 
    the masked arrays containing only the noticed bins. 

    Attributes:
    -----------    
    energs: np.array(float)
        The array of physical photon energies over which the model is computed. 
        Defined as the middle of each bin in the energy range stored in the 
        instrument response provided.    
        
    energ_bounds: np.array(float)
        The array of energy bin widths, for each bin over which the model is 
        computed. Defined as the difference between the uppoer and lower bounds 
        of the energy bins stored in the insrument response provided. 
               
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
        
    n_chans: int 
        The number of channels that are to be noticed during the fit.
        
    _all_chans: int 
        The total number of channels in the loaded response matrix.
        
    n_bins: int 
        Only used for two-dimensional data fitting. Defined as the number of 
        noticed channels, times the number of bins in the second dimension 
        (e.g. Fourier frequency).
        
    _all_bins: int 
        Only used for two-dimensional data fitting. Defined as the total number 
        of  channels, times the number of bins in the second dimension 
        (e.g. Fourier frequency).
                
    _emin_unmasked, _emax_unmasked, _ebounds_unmasked, _ewidths_unmasked: np.array(float)
        The array of every lower bound, upper bound, channel center and channel 
        widths stored in the response, regardless of which ones are ignored or 
        noticed during the fit. Used exclusively to facilitate book-keeping 
        internal to the fitter class. 
        
    _data_unmasked, _data_err_unmasked: np.array(float)
        The array of every cout rate and relative error contained in the 
        spectrum, regardless of which ones are ignored or noticed during the 
        fit. Used exclusively to facilitate book-keeping internal to the fitter
        class.   
    """
    
    def __init__(self):   
        self.energs = 0.5*(self.response.energ_hi+self.response.energ_lo)
        self.energ_bounds = self.response.energ_hi-self.response.energ_lo
        self.ebounds = 0.5*(self.response.emax+self.response.emin)
        self.ewidths = self.response.emax - self.response.emin
        self.ebounds_mask = np.full((self.response.n_chans), True)
        pass

    def _set_unmasked_data(self,extra_dim_size=1.):
        """
        This initializer method is used to set up the unmasked arrays for later 
        book-keeping. Classes inheriting from EnergyDependentFit should call it 
        immediately after setting the data and energy/channel arrays. 
        
        Parameters:
        -----------
        extra_dim_size: int, default=1
            The dimension of the data in the direction in addition to photon energy 
            (e.g., the number of Fourier frequency bins). Necessary to store the 
            total number of data bins loaded. 
        """
        
        self._emin_unmasked = self.response.emin
        self._emax_unmasked = self.response.emax
        self._ebounds_unmasked = self.ebounds
        self._ewidths_unmasked = self.ewidths
        self._data_unmasked = self.data
        self._data_err_unmasked = self.data_err

        if self.twod_data is True:
            self._all_chans = self._ebounds_unmasked.size
            self._all_bins = extra_dim_size*self._all_chans
            self.n_chans = self._all_chans
            self.n_bins = self._all_bins
        return
        
    def ignore_energies(self,bound_lo,bound_hi):
        """
        This method Aadjusts the arrays stored such that they (and the fit) 
        ignore selected channels based on their energy bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored energy interval.
        bound_hi : float
            Higher bound of ignored energy interval.    
        """
        
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Energy bounds must be floats or integers")
        
        self.ebounds_mask = ((self._emin_unmasked<bound_lo)|
                             (self._emax_unmasked>bound_hi))&self.ebounds_mask
       
        #take the unmasked arrays and keep only the bounds we want
        self.emin = np.extract(self.ebounds_mask,self._emin_unmasked)
        self.emax = np.extract(self.ebounds_mask,self._emax_unmasked)
        self.ebounds = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        self.ewidths = np.extract(self.ebounds_mask,self._ewidths_unmasked)   
        self.n_chans = self.ebounds_mask[self.ebounds_mask==True].size

        #filter 2d data is more complex because we have to filter row by row 
        #or column by column, depending on the format 
        if self.twod_data is True:  
            self.n_bins = self.n_chans*self.n_freqs
            if self.units != "lags":
                data_filter_first_dim = self._filter_2d_by_mask(
                                        self._data_unmasked[:self._all_bins],
                                        self.ebounds_mask
                                        )
                error_filter_first_dim = self._filter_2d_by_mask(
                                         self._data_err_unmasked[:self._all_bins],
                                         self.ebounds_mask
                                         )              
                data_filter_second_dim = self._filter_2d_by_mask(
                                         self._data_unmasked[self._all_bins:],
                                         self.ebounds_mask
                                         )
                error_filter_second_dim = self._filter_2d_by_mask(
                                          self._data_err_unmasked[self._all_bins:],
                                          self.ebounds_mask
                                          )                
                self.data = np.append(data_filter_first_dim,data_filter_second_dim)
                self.data_err = np.append(error_filter_first_dim,error_filter_second_dim)              
            else:
                self.data = self._filter_2d_by_mask(
                            self._data_unmasked,
                            self.ebounds_mask
                            )
                self.data_err = self._filter_2d_by_mask(
                                self._data_err_unmasked,
                                self.ebounds_mask
                                )
        else: 
            self.data = np.extract(self.ebounds_mask,self._data_unmasked)
            self.data_err = np.extract(self.ebounds_mask,self._data_err_unmasked)            
        return
   
    def notice_energies(self,bound_lo,bound_hi):
        """
        This method adjusts the data arrays stored such that they (and the fit) 
        notice selected (previously ignore) channels  based on their energy 
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
              
        #if bounds of channel lie in noticed energies, notice channel
        self.ebounds_mask = self.ebounds_mask|np.logical_not(
                            (self._emin_unmasked<bound_lo)|
                            (self._emax_unmasked>bound_hi))

        #take the unmasked arrays and keep only the bounds we want
        self.emin = np.extract(self.ebounds_mask,self._emin_unmasked)
        self.emax = np.extract(self.ebounds_mask,self._emax_unmasked)
        self.ebounds = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        self.ewidths = np.extract(self.ebounds_mask,self._ewidths_unmasked)   
        self.n_chans = self.ebounds_mask[self.ebounds_mask==True].size        

        #filter 2d data is more complex because we have to filter row by row 
        #or column by column, depending on the format 
        if self.twod_data is True:
            self.n_bins = self.n_chans*self.n_freqs
            if self.units != "lags":
                data_filter_first_dim = self._filter_2d_by_mask(
                                        self._data_unmasked[:self._all_bins],
                                        self.ebounds_mask
                                        )
                error_filter_first_dim = self._filter_2d_by_mask(
                                         self._data_err_unmasked[:self._all_bins],
                                         self.ebounds_mask
                                         )              
                data_filter_second_dim = self._filter_2d_by_mask(
                                         self._data_unmasked[self._all_bins:],
                                         self.ebounds_mask
                                         )
                error_filter_second_dim = self._filter_2d_by_mask(
                                          self._data_err_unmasked[self._all_bins:],
                                          self.ebounds_mask
                                          )                
                self.data = np.append(data_filter_first_dim,data_filter_second_dim)
                self.data_err = np.append(error_filter_first_dim,error_filter_second_dim)              
            else:
                self.data = self._filter_2d_by_mask(
                            self._data_unmasked,
                            self.ebounds_mask
                            )
                self.data_err = self._filter_2d_by_mask(
                                self._data_err_unmasked,
                                self.ebounds_mask
                                )
        else:
            self.data = np.extract(self.ebounds_mask,self._data_unmasked)
            self.data_err = np.extract(self.ebounds_mask,self._data_err_unmasked)               
        return

    def _filter_2d_by_mask(self,arr,mask):
        """
        This method filters either the rows or columns of a two-dimensional 
        array, depending on the dependence of the data products used. Currently 
        the method assumes that the input data is a function of Fourier frequency 
        and energy. For example, one could input lag-frequency spectra, or 
        energy-covariance, or residuals for an appropriate two-dimensional model.
        
        Parameters:
        -----------
        arr: np.array(float,float)  
            The two-dimensional array to be filtered 
        mask: np.array(bool)
            The mask to be applied to the array - elements labelled as True in 
            the mask are kept, ones labelled as False are filtered out. 
            
        Returns:
        --------
        filtered_array: np.array(float,float)
            The input two-d array, reduced and filtered to include only the 
            noticed energy channels 
        """    
        
        filtered_array = [] 
        if self.dependence == "energy":
            arr_reshape = arr.reshape((self.n_freqs,self._all_chans))
            for i in range(self.n_freqs):
                extract_row = np.extract(self.ebounds_mask,arr_reshape[i,:])
                filtered_array = np.append(filtered_array,extract_row)  
        elif self.dependence == "frequency":
            arr_reshape = arr.reshape((self._all_chans,self.n_freqs))
            filtered_array = arr_reshape[mask,:]
            filtered_array = filtered_array.reshape(self.n_bins)
        return filtered_array

class FitPowerSpectrum(SimpleFit):
    """
    Least-chi squared fitter class for a powerspectrum, defined as the product 
    between a Fourier-transformed lightcurve and its complex conjugate. 
    
    Given an array of Fourier frequencies, a power spectrum, its error and a 
    model (defined in Fourier space), this class handles fitting internally 
    using the lmfit library.    
    
    Poisson noise in the data is not accounted for explicitely. Users should 
    either pass noise-subtracted data, or include a constant component in the 
    model definition (see below).   
        
    Attributes inherited from SimpleFit:
    ------------------------------------
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
        An array storing the data to be fitted. If the data is complex and/or 
        multi-dimensional, it is flattened to a single dimension in order to be 
        compatible with the LMFit fitter methods.
   
    data_err: np.array(float)
        An array containing the uncertainty on the data to be fitted. It is also 
        stored as a one-dimensional array regardless of the type or dimensionality 
        of the initial data.   
        
    Other attributes:
    -----------------    
    freqs: np.array(float)
        The Fourier frequency over which both the data and model are defined, 
        in units of Hz.           
    """ 

    def __init__(self):
        SimpleFit.__init__(self)
        self.freqs = None 
        self.twod_data = False
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
    
    def _minimizer(self,params):
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
            model = self.eval_model(params=self.model_params)
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
            comps = LM_result(model=self.model,params=self.model_params)
            comps = comps.eval_components(freq=self.freqs)
            for key in comps.keys():
                if units == "power":                
                    ax1.plot(self.freqs,comps[key],label=key,lw=2)
                elif units == "fpower":
                    ax1.plot(self.freqs,comps[key]*self.freqs,label=key,lw=2)
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

class FitTimeAvgSpectrum(SimpleFit,EnergyDependentFit):
    """
    Least-chi squared fitter class for a time averaged spectrum, defined as the  
    count rate as a function of photon channel energy bound. 
    
    Given an instrument response, a count rate spectrum, its error and a 
    model (defined in energy space), this class handles fitting internally 
    using the lmfit library.    
        
    Attributes inherited from SimpleFit:
    ------------------------------------
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
        An array storing the data to be fitted. If the data is complex and/or 
        multi-dimensional, it is flattened to a single dimension in order to be 
        compatible with the LMFit fitter methods.
   
    data_err: np.array(float)
        An array containing the uncertainty on the data to be fitted. It is also 
        stored as a one-dimensional array regardless of the type or dimensionality 
        of the initial data.       
    
    Attributes inherited from EnergyDependentFit:
    ---------------------------------------------    
    energs: np.array(float)
        The array of physical photon energies over which the model is computed. 
        Defined as the middle of each bin in the energy range stored in the 
        instrument response provided.    
        
    energ_bounds: np.array(float)
        The array of energy bin widths, for each bin over which the model is 
        computed. Defined as the difference between the uppoer and lower bounds 
        of the energy bins stored in the insrument response provided. 
               
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
        
    n_chans: int 
        The number of channels that are to be noticed during the fit.
        
    _all_chans: int 
        The total number of channels in the loaded response matrix.
        
    n_bins: int 
        Only used for two-dimensional data fitting. Defined as the number of 
        noticed channels, times the number of bins in the second dimension 
        (e.g. Fourier frequency).
        
    _all_bins: int 
        Only used for two-dimensional data fitting. Defined as the total number 
        of  channels, times the number of bins in the second dimension 
        (e.g. Fourier frequency).
                
    _emin_unmasked, _emax_unmasked, _ebounds_unmasked, _ewidths_unmasked: np.array(float)
        The array of every lower bound, upper bound, channel center and channel 
        widths stored in the response, regardless of which ones are ignored or 
        noticed during the fit. Used exclusively to facilitate book-keeping 
        internal to the fitter class. 
        
    _data_unmasked, _data_err_unmasked: np.array(float)
        The array of every cout rate and relative error contained in the 
        spectrum, regardless of which ones are ignored or noticed during the 
        fit. Used exclusively to facilitate book-keeping internal to the fitter
        class.       
    
    Other attributes:
    -----------------
    response: nDspec.ResponseMatrix
        The instrument response matrix corresponding to the spectrum to be 
        fitted. It is required to define the energy grids over which model and
        data are defined.   
    """ 
    
    def __init__(self):
        SimpleFit.__init__(self)
        self.twod_data = False
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
        EnergyDependentFit.__init__(self)  
        #this loads the spectrum in units of counts/s/keV
        self.data = counts/exposure/self.ewidths
        self.data_err = error/exposure/self.ewidths
        self._set_unmasked_data()
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

    def _minimizer(self,params):
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
        model_fold = self.eval_model(params=params,energ=self.energs)
        if units == "data":   
            model = np.extract(self.ebounds_mask,model_fold)   
            ylabel = "Folded counts/s/keV"
        elif units.count("unfold"):
            power = units.count("e") 
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

class FitCrossSpectrum(SimpleFit,EnergyDependentFit):
    
    def __init__(self):
        SimpleFit.__init__(self)
        self.twod_data = True 
        self.ref_band = None
        self.freqs = None 
        self._times = None
        self.crossspec = None
        self._supported_coordinates = ["cartesian","polar","lags"]
        self._supported_models = ["irf","transfer","cross"]
        self._supported_products = ["frequency","energy"]
        self.renorm_phase = False
        self.renorm_modulus = False
        pass

    def set_product_dependence(self,depend):
        #dep can only be frequency or energy for now, eventually polarimetry
        if depend not in self._supported_products:
            raise AttributeError("Unsopprted products for the cross spectrum")
        else:
            self.dependence = depend
        return 

    def set_coordinates(self,units="cartesian"):    
        if units not in self._supported_coordinates:
            raise AttributeError("Unsopprted units for the cross spectrum")
        else:
            self.units = units
        return 

    #sub_bounds is the bounds with the subjugate channels (will need to be
    #extended to the lowest bins)
    #freqs is the internal frequency grid where we evaluate the model
    #freq_bounds are the bounds over which we frequency-average to get
    #energy dependent products
    #explicitely show in the documentation that there are many ways to build
    #data and users have a lot of freedom.
    def set_data(self,response,ref_bounds,sub_bounds,data,
                 data_err=None,freq_grid=None,time_grid=None,
                 freq_bins=None,time_res=None,seg_size=None,norm=None):
        #I need a safeguard to put in the lowest+highest channels in the matrix
        if self.units is None:
            raise AttributeError("Cross spectrum units not defined") 
        if norm is None:
            norm = "abs"
        #combine the edges of the reference and subject bands with those of the matrix
        #then sort+keep only the ones that are not repeated, and rebin the matrix
        #to this grid of channels
        rebin_bounds = np.append(sub_bounds,ref_bounds).reshape(len(sub_bounds)+len(ref_bounds))
        rebin_bounds = np.append(rebin_bounds,response.emin[0])
        rebin_bounds = np.append(rebin_bounds,response.emax[-1])
        rebin_bounds = np.unique(np.sort(rebin_bounds))
        bounds_lo = rebin_bounds[:-1]
        bounds_hi = rebin_bounds[1:] 
        self.response = response.rebin_channels(bounds_lo,bounds_hi) 
        self.ref_band = ref_bounds
        EnergyDependentFit.__init__(self)  
        self.n_chans = self.ebounds_mask[self.ebounds_mask==True].size
        
        if self.dependence == "frequency":
            self._freq_dependent_cross(data,data_err,freq_grid,time_grid,time_res,seg_size,norm)
        elif self.dependence == "energy":
            self._energ_dependent_cross(freq_bins,data,data_err,freq_grid,time_grid,time_res,seg_size)
        else:
            print("error")    
        self._set_unmasked_data(self.n_freqs)
        return

    def _freq_dependent_cross(self,data,data_err=None,
                              freq_grid=None,time_grid=None,
                              time_res=None,seg_size=None,norm=None):
        
        if getattr(data, '__module__', None) == "stingray.events":
            #check here that timeres, seg size and norm are all defined
            events_ref = data.filter_energy_range(self.ref_band)
            ps_ref = AveragedPowerspectrum.from_events(events_ref,
                                                       segment_size=seg_size,
                                                       dt=time_res,norm=norm,silent=True)
            ctrate_ref = get_average_ctrate(events_ref.time,events_ref.gti,seg_size)
            noise_ref = poisson_level(norm=norm, meanrate=ctrate_ref)     
                
            #set the (linearly spaced) internal time and frequency grids
            lc_length = ps_ref.n*time_res
            time_samples = int(lc_length/time_res)
            self._times = np.linspace(time_res,lc_length,time_samples)
            self.freqs = np.array(ps_ref.freq)

            #loop over all channels of interest and get the desired
            #timing products
            self.data = []
            self.data_err = []
            
            for i in range(self.n_chans):
                events_sub = events.filter_energy_range([self.response.emin[i],self.response.emax[i]])
                #get the cross spectrum
                cs = AveragedCrossspectrum.from_events(events_sub,events_ref,
                                                       segment_size=seg_size,dt=time_res,norm=norm,silent=True)
                if self.units == "lags":
                    lag, lag_err = cs.time_lag() 
                    self.data = np.append(self.data,lag)
                    self.data_err = np.append(self.data_err,lag_err)
                else:
                    ps_sub = AveragedPowerspectrum.from_events(events_sub,
                                                               segment_size=seg_size,
                                                               dt=time_res,norm=norm,silent=True)    
                    ctrate_sub = get_average_ctrate(events_sub.time,events_sub.gti,seg_size)                    
                    noise_sub = poisson_level(norm=norm, meanrate=ctrate_sub)                      
                    data_size = len(cs.freq)
                    
                    if norm == "frac":
                        N = 2./noise_ref 
                    elif norm == "abs":
                        N = 2.*noise_sub
                    else:
                        print("Normalization is wrong")
                        N = 1.
                
                    if self.units == "cartesian":    
                        data_first_dim = np.real(cs.power)
                        data_second_dim = np.imag(cs.power)                
                        error_first_dim = np.sqrt((ps_sub.power*ps_ref.power+ \
                                                   np.real(cs.power)**2- \
                                                   np.imag(cs.power)**2)/(2.*N))
                        error_second_dim = np.sqrt((ps_sub.power*ps_ref.power- \
                                                    np.real(cs.power)**2+ \
                                                    np.imag(cs.power)**2)/(2.*N))
                    elif self.units == "polar":
                        data_first_dim = np.absolute(cs.power)
                        error_first_dim = np.sqrt(ps_sub.power*ps_ref.power/(2.*N))
                        data_second_dim, error_second_dim = cs.phase_lag()
                    
                    if i == 0:
                        self.data = np.append(self.data,data_first_dim)
                        self.data_err = np.append(self.data_err,error_first_dim)
                    else:
                        self.data = np.insert(self.data,i*data_size,data_first_dim)
                        self.data_err = np.insert(self.data_err,i*data_size,error_first_dim) 
                    self.data = np.append(self.data,data_second_dim)
                    self.data_err = np.append(self.data_err,error_second_dim)

            #here we just pass the data+grids by hand
        else:
            #we can explicitely pass frequency and time grids
            if (time_grid is not None and freq_grid is not None):
                self._times = time_grid
                self.freqs = freq_grid
            #or we can explicitel pass a frequency grid alone, and the time grid is 
            #reconstructed automatically 
            elif freq_grid is not None:
                self.freqs = freq_grid
                #now revert the grid from frequency to time, and save in times
                #this is needed to allocate the ndspec objects
                time_res = 0.5/(self.freqs[-1]+self.freqs[0])
                lc_length = (self.freqs.size+1)*2*time_res
                time_samples = int(lc_length/time_res)
                #check the spacing of the frequency array, allocate time array accordingly
                if (np.allclose(self.freqs, self.freqs[0]) is False):
                    self._times = np.geomspace(time_res,lc_length,time_samples)
                else:
                    self._times = np.linspace(time_res,lc_length,time_samples)              
            else:
                print("Frequency and/or time grids undefined")
            #the reason setting the grids is flexible is to allow users to avoid numerical
            #issues due to the discrete FFT at the highest and/or lowest frequency bins
            
            #if everything is ok, set the data. 
            self.data = data
            self.data_err = data_err  
        
        self.n_freqs = self.freqs.size
        return

    def _energ_dependent_cross(self,freq_bounds,data,data_err=None,
                               freq_grid=None,time_grid=None,
                               time_res=None,seg_size=None):

        self.freq_bounds = freq_bounds
        self.n_freqs = self.freq_bounds.size-1
                
        self.data = data
        self.data_err = data_err

        #we can explicitely pass frequency and time grids
        if (time_grid is not None )&(freq_grid is not None):
            self._times = time_grid
            self.freqs = freq_grid
        #or do so implicetly with a segment size and time resolution
        elif (time_res is not None)&(seg_size is not None):
            freqs = fftfreq(int(seg_size/time_res),time_res)
            self.freqs = freqs[freqs>0]        
            lc_length = (self.freqs.size+1)*2*time_res
            time_samples = int(lc_length/time_res)
            self._times = np.linspace(time_res,lc_length,time_samples)
        #or we can explicitel pass a frequency grid alone, and the time grid is 
        #reconstructed automatically 
        elif freq_grid is not None:
            self.freqs = freq_grid
            #now revert the grid from frequency to time, and save in times
            #this is needed to allocate the ndspec objects
            time_res = 0.5/(self.freqs[-1]+self.freqs[0])
            lc_length = (self.freqs.size+1)*2*time_res
            time_samples = int(lc_length/time_res)
            #check the spacing of the frequency array, allocate time array accordingly
            if (np.allclose(self.freqs, self.freqs[0]) is False):
                self._times = np.geomspace(time_res,lc_length,time_samples)
            else:
                self._times = np.linspace(time_res,lc_length,time_samples)              
        else:
            print("Frequency and/or time grids undefined")    
        #the reason setting the grids is flexible is to allow users to avoid numerical
        #issues due to the discrete FFT at the highest and/or lowest frequency bins        
        return 

    def set_model(self,model,model_type="irf",params=None):
        if model_type not in self._supported_models:
            raise AttributeError("Unsopprted model type")  
        self.model_type = model_type
        self.crossspec = CrossSpectrum(self._times,freqs=self.freqs,energ=self.energs)
        self.model = model 
        if params is None:
            self.model_params = self.model.make_params(verbose=True)
        else:
            self.model_params = params
        return 
        
    def set_psd_weights(self,psd_weights):       
        if self.model_type != "cross":
            self.crossspec.set_psd_weights(psd_weights)
        else:
            print("Power spectrum weight not needed")
        return 
    
    def eval_model(self,params=None,ref_band=None,mask=True):
        if ref_band is None:
            ref_band = self.ref_band
                
        #evaluate the model for the chosen parameters
        if params is None:
            params= self.model_params
        model_eval = self.model.eval(params,freqs=self.freqs,energs=self.energs,times=self._times)
        #store the model in the cross spectrum, depending on the type
        if self.model_type == "irf":
            self.crossspec.set_impulse(np.transpose(model_eval))
            self.crossspec.set_reference_energ(self.ref_band)
            self.crossspec.cross_from_irf()
        elif self.model_type == "transfer":
            self.crossspec.set_transfer(np.transpose(model_eval))
            self.crossspec.set_reference_energ(self.ref_band)
            self.crossspec.cross_from_transfer()
        elif self.model_type == "cross":
            #transposing is required to ensure the units are correct 
            self.crossspec.cross = np.transpose(model_eval)
            
        #fold the instrument response:
        folded_eval = self.response.convolve_response(self.crossspec,units_in="rate",units_out="channel")  

        #return the appropriately structured products
        #filtering may be necessary ugh
        if self.dependence == "frequency":
            eval = self._freq_dependent_model(folded_eval)
        elif self.dependence == "energy":
            eval = self._energ_dependent_model(folded_eval,params)
        else:
            print("error")  

        #tbd: only do this if some elements in ebounds_mask are False
        #make sure that this works for freq dependent stuff 
        #also check that the synthax/wording is consistent with the other classes
        if mask is True:
            if self.units != "lags":
                model_first_dim = self._filter_2d_by_mask(eval[:self._all_bins],self.ebounds_mask)
                model_second_dim = self._filter_2d_by_mask(eval[self._all_bins:],self.ebounds_mask)
                eval = np.append(model_first_dim,model_second_dim)
            else:
                eval = self._filter_2d_by_mask(eval,self.ebounds_mask)
        return eval

    def _freq_dependent_model(self,folded_eval):
        model = []
        sub_bounds = np.array([self._ebounds_unmasked-0.5*self._ewidths_unmasked,
                               self._ebounds_unmasked+0.5*self._ewidths_unmasked])
        sub_bounds = np.transpose(sub_bounds)
        if self.units == "lags":
            for i in range(self._all_chans):
                model_eval = folded_eval.lag_frequency(sub_bounds[i])
                model = np.append(model,model_eval)
        elif self.units == "cartesian":
            real = []
            imag = []             
            for i in range(self._all_chans):
                real_eval = folded_eval.real_frequency(sub_bounds[i])
                imag_eval = folded_eval.imag_frequency(sub_bounds[i])            
                real = np.append(real,real_eval)
                imag = np.append(imag,imag_eval)
            model = np.concatenate((real,imag))
        elif self.units == "polar":
            mod = []
            phase = []            
            for i in range(self._all_chans):
                mod_eval = folded_eval.mod_frequency(sub_bounds[i])
                phase_eval = folded_eval.phase_frequency(sub_bounds[i])         
                mod = np.append(mod,mod_eval)
                phase = np.append(phase,phase_eval)            
            model = np.concatenate((mod,phase))
        else:
            print("error")    
        return model

    def _energ_dependent_model(self,folded_eval,params):
        model = []
        if self.units == "lags":   
            for i in range(self.n_freqs):
                f_mean = 0.5*(self.freq_bounds[1:]+self.freq_bounds[:-1])
                if self.renorm_phase is True:
                    par_key = 'phase_renorm_'+str(i+1)
                    phase_pars = LM_Parameters()
                    phase_pars.add('renorm',value=params[par_key].value,
                                   min=params[par_key].min,max=params[par_key].max,
                                   vary=params[par_key].vary)
                    
                    phase_model = folded_eval.phase_energy([self.freq_bounds[i],self.freq_bounds[i+1]])
                    model_eval = self.phase_renorm_model.eval(phase_pars,array=phase_model)/(2*np.pi*f_mean[i])  
                else:
                    model_eval = folded_eval.lag_energy([self.freq_bounds[i],self.freq_bounds[i+1]])
                model = np.append(model,model_eval)
        elif self.units == "cartesian":
            real = []
            imag = [] 
            for i in range(self.n_freqs):
                real_eval = folded_eval.real_energy([self.freq_bounds[i],self.freq_bounds[i+1]])
                imag_eval = folded_eval.imag_energy([self.freq_bounds[i],self.freq_bounds[i+1]])
                if self.renorm_modulus is True:
                    par_key = 'mods_renorm_'+str(i+1)
                    mods_pars = LM_Parameters()
                    mods_pars.add('renorm',value=params[par_key].value,
                                   min=params[par_key].min,max=params[par_key].max,
                                   vary=params[par_key].vary)                                       
                    real_eval = self.mod_renorm_model.eval(mods_pars,array=real_eval)                                     
                    imag_eval = self.mod_renorm_model.eval(mods_pars,array=imag_eval)
                if self.renorm_phase is True:
                    par_key = 'phase_renorm_'+str(i+1)
                    angle = self.model_params[par_key].value
                    real_eval = np.cos(angle)*real_eval - np.sin(angle)*imag_eval
                    imag_eval = np.cos(angle)*imag_eval + np.sin(angle)*real_eval
                real = np.append(real,real_eval)
                imag = np.append(imag,imag_eval)
            model = np.concatenate((real,imag))
        elif self.units == "polar":
            mod = []
            phase = []
            for i in range(self.n_freqs):
                mod_model = folded_eval.mod_energy([self.freq_bounds[i],self.freq_bounds[i+1]])
                phase_model = folded_eval.phase_energy([self.freq_bounds[i],self.freq_bounds[i+1]])
                if self.renorm_modulus is True:
                    par_key = 'mods_renorm_'+str(i+1)
                    mods_pars = LM_Parameters()
                    mods_pars.add('renorm',value=params[par_key].value,
                                   min=params[par_key].min,max=params[par_key].max,
                                   vary=params[par_key].vary)             
                    mod_model = self.mod_renorm_model.eval(mods_pars,array=mod_model)                     
                if self.renorm_phase is True:                    
                    par_key = 'phase_renorm_'+str(i+1)
                    phase_pars = LM_Parameters()
                    phase_pars.add('renorm',value=params[par_key].value,
                                   min=params[par_key].min,max=params[par_key].max,
                                   vary=params[par_key].vary)
                    phase_model = self.phase_renorm_model.eval(phase_pars,array=phase_model)                   
                mod = np.append(mod,mod_model)
                phase = np.append(phase,phase_model)
            model = np.concatenate((mod,phase))
        else:
            print("weird units raise proper error tbd")            
        return model

    def renorm_phases(self,value):
        #add complaint if people activate this for freq dependency
        self.renorm_phase = value
        if self.renorm_phase is True:
            self.phase_renorm_model = LM_Model(self._renorm_phase)
            phase_pars = LM_Parameters()
            for index in range(self.n_freqs):   
                phase_pars.add('phase_renorm_'+str(index+1), value=0,min=-0.2,max=0.2,vary=True)            
            self.model_params = self.model_params + phase_pars
        return
        
    def _renorm_phase(self,array,renorm):
        return array + renorm

    def renorm_mods(self,value):
        #add complaint if people activate this for freq dependency
        self.renorm_modulus = value
        if self. renorm_modulus is True:
            self.mod_renorm_model = LM_Model(self._renorm_modulus)
            mods_pars = LM_Parameters()
            for index in range(self.n_freqs):   
                mods_pars.add('mods_renorm_'+str(index+1), value=1,min=0,max=1e5,vary=True)            
            self.model_params = self.model_params + mods_pars            
        return

    def _renorm_modulus(self,array,renorm):
        return renorm*array

    def _minimizer(self,params):
        if self.likelihood is None:
            model = self.eval_model(params,ref_band=self.ref_band)
            residuals = (self.data-model)/self.data_err
        else:
            raise TypeError("custom likelihood not implemented yet")
        return residuals
    
    def plot_data_1d(self,return_plot=False):
        #depending on the units of the data, we need to set the number of 
        #spectra that were loaded, what goes on the x axis, and the bounds
        #of where one spectrum ends and the next begins
        if self.dependence == "frequency":
            x_axis = self.freqs
            x_axis_label = "Frequency (Hz)"
            #here we need to look at the edges of each bin, NOT at the center
            #which is contained in ebounds
            #this is wrong, it needs to be the noticed channel bounds blergh
            bounds_min = self.ebounds-0.5*self.ewidths
            channel_bounds = np.append(bounds_min,self.ebounds[-1]+0.5*self.ewidths[-1])
            labels = np.round(channel_bounds,1)
            units = "keV"
            spec_number = self.n_chans
            data_bound = self.n_freqs
        elif self.dependence == "energy":
            x_axis = self.ebounds
            x_axis_label = "Energy (keV)"
            labels = np.round(self.freq_bounds,1)
            units = "Hz"
            spec_number = self.n_freqs    
            data_bound = self.n_chans
           
        if self.units != "lags":
            fig, ((ax1),(ax2)) = plt.subplots(1,2,figsize=(12.,5.))  
            
            ax2.hlines(0,x_axis[0],x_axis[-1],color='black',ls=':',zorder=3)
           
            if self.units == "cartesian":
                left_label = "Real part"
                right_label = "Imaginary part"                
            elif self.units == "polar":
                left_label = "Modulus"
                right_label = "Phase" 
                
            for i in range(spec_number):
                ax1.errorbar(x_axis,self.data[i*data_bound:(i+1)*data_bound],
                             yerr=self.data_err[i*data_bound:(i+1)*data_bound],
                             marker='o',linestyle='',#drawstyle="steps-mid",
                             label=f"{labels[i]}-{labels[i+1]} {units}")
                ax2.errorbar(x_axis,self.data[self.n_bins+i*data_bound:self.n_bins+(i+1)*data_bound],
                             yerr=self.data_err[self.n_bins+i*data_bound:self.n_bins+(i+1)*data_bound],
                             marker='o',linestyle='',#drawstyle="steps-mid",
                             label=f"{labels[i]}-{labels[i+1]} {units}") 
                
            ax1.set_yscale("log")
            ax1.set_xscale("log")
            ax1.set_xlabel(x_axis_label)
            ax1.set_ylabel(left_label)
            ax2.set_xscale("log")
            ax2.set_xlabel(x_axis_label)
            ax2.set_ylabel(right_label)
            ax2.legend(loc="best",ncol=2)
        else:            
            fig, ((ax1)) = plt.subplots(1,1,figsize=(6.,4.5)) 
            
            ax1.hlines(0,x_axis[0],x_axis[-1],color='black',ls=':',zorder=3)
            
            for i in range(spec_number):
                ax1.errorbar(x_axis,self.data[i*data_bound:(i+1)*data_bound],
                             yerr=self.data_err[i*data_bound:(i+1)*data_bound],
                             marker='o',linestyle='',#drawstyle="steps-mid",
                             label=f"{labels[i]}-{labels[i+1]} {units}") 
            
            ax1.set_xscale("log")
            ax1.legend(loc="best",ncol=2,fontsize=12)
            ax1.set_ylabel("Lag (s)")
            ax1.set_xlabel(x_axis_label)
        
        fig.tight_layout()            
        if return_plot is True:
            return fig
        else:
            return 

    def plot_data_2d(self,use_phase=False,return_plot=False):        
        #depending on the units of the data, we need to set the number of 
        #spectra that were loaded, what goes on the x axis, and the bounds
        #of where one spectrum ends and the next begins

        if self.dependence == "frequency":
            x_axis = self.freqs
            y_axis = self._ebounds_unmasked
            #here we need to look at the edges of each bin, NOT at the center
            #which is contained in ebounds
            spec_number = self.n_chans
            data_bound = self.n_freqs
        elif self.dependence == "energy":
            x_axis =  0.5*(self.freq_bounds[1:]+self.freq_bounds[:-1])
            y_axis = self._ebounds_unmasked
            spec_number = self.n_freqs    
            data_bound = self.n_chans        
        
        if self.units != "lags":
            if self.dependence=="energy":
                mask_twod = np.tile(self.ebounds_mask,self.n_freqs)
                left_data = self._data_unmasked[:self._all_bins].reshape((self.n_freqs,self._all_chans))
                right_data = self._data_unmasked[self._all_bins:].reshape((self.n_freqs,self._all_chans))
            elif self.dependence=="frequency":
                mask_twod = np.transpose(np.tile(self.ebounds_mask,self.n_freqs))
                left_data = np.transpose(self._data_unmasked[:self._all_bins].reshape((self._all_chans,self.n_freqs)))
                right_data = np.transpose(self._data_unmasked[self._all_bins:].reshape((self._all_chans,self.n_freqs)))                
            mask_twod = mask_twod.reshape((self.n_freqs,self._all_chans))
            mask_twod = np.logical_not(mask_twod) 
            left_data = np.transpose(np.ma.masked_where(mask_twod, left_data))
            right_data = np.transpose(np.ma.masked_where(mask_twod, right_data))

            fig, ((ax1),(ax2)) = plt.subplots(1,2,figsize=(12.,5.)) 
            if self.units == "polar":
                left_plot = ax1.pcolormesh(x_axis,y_axis,np.log10(left_data),cmap="viridis",
                                           shading='auto',linewidth=0)
                color_min = np.min([np.min(right_data),-0.01])
                color_max = np.max([np.max(right_data),0.01])
                phase_norm = TwoSlopeNorm(vmin=color_min,vcenter=0,vmax=color_max) 
                right_plot = ax2.pcolormesh(x_axis,y_axis,right_data,cmap="BrBG",
                                            shading='auto',linewidth=0,norm=phase_norm)

                ax1.set_title("log10(Modulus)")
                ax2.set_title("Phase")
            else:
                left_plot = ax1.pcolormesh(x_axis,y_axis,np.log10(left_data),cmap="viridis",
                                           shading='auto',linewidth=0)
                right_plot = ax2.pcolormesh(x_axis,y_axis,right_data,cmap="cividis",
                                            shading='auto',linewidth=0)
                ax1.set_title("Real part")
                ax2.set_title("Imaginary part")                
            fig.colorbar(left_plot, ax=ax1)
            fig.colorbar(right_plot, ax=ax2)
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ymin = np.max([self.ebounds[0]-0.5*self.ewidths[0],1e-1])
            ymax = self.ebounds[-1]+0.5*self.ewidths[-1]
            ax1.set_ylim([ymin,ymax])
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Energy (keV)")

            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_ylim([ymin,ymax])
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Energy (keV)")
        else:   
            fig, ((ax1)) = plt.subplots(1,1,figsize=(6.,4.5))
            if self.dependence == "energy":               
                mask_twod = np.tile(self.ebounds_mask,self.n_freqs)
                plot_data = self._data_unmasked.reshape((self.n_freqs,self._all_chans))
            elif self.dependence == "frequency":
                mask_twod = np.transpose(np.tile(self.ebounds_mask,self.n_freqs))
                plot_data = np.transpose(self._data_unmasked.reshape((self._all_chans,self.n_freqs)))
            mask_twod = mask_twod.reshape((self.n_freqs,self._all_chans))
            mask_twod = np.logical_not(mask_twod) 
            if use_phase is True:
                plot_data = plot_data*(2.*np.pi*x_axis.reshape(self.n_freqs,1))
            plot_data = np.transpose(np.ma.masked_where(mask_twod, plot_data))
            color_min = np.min([np.min(plot_data),-0.01])
            color_max = np.max([np.max(plot_data),0.01])
            lag_norm = TwoSlopeNorm(vmin=color_min,vcenter=0,vmax=color_max) 
            lag_plot = ax1.pcolormesh(x_axis,y_axis,plot_data,cmap="BrBG",
                                      shading='auto',linewidth=0,norm=lag_norm)
            if use_phase is False:
                ax1.set_title("Lag (s)")        
            else:
                ax1.set_title("Phase")   
            fig.colorbar(lag_plot, ax=ax1)
            ymin = np.max([self.ebounds[0]-0.5*self.ewidths[0],1e-1])
            ymax = self.ebounds[-1]+0.5*self.ewidths[-1]
            ax1.set_ylim([ymin,ymax])
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Energy (keV)")
        
        fig.tight_layout()
        if return_plot is True:
            return fig
        else:
            return      

    def plot_model_1d(self,plot_data=True,params=None,residuals="delchi",return_plot=False):
        
        if self.dependence == "frequency":
            x_axis = self.freqs
            x_axis_label = "Frequency (Hz)"
            #here we need to look at the edges of each bin, NOT at the center
            #which is contained in ebounds
            #this is wrong, it needs to be the noticed channel bounds blergh
            bounds_min = self.ebounds-0.5*self.ewidths
            channel_bounds = np.append(bounds_min,self.ebounds[-1]+0.5*self.ewidths[-1])
            labels = np.round(channel_bounds,1)
            units = "keV"
            spec_number = self.n_chans
            data_bound = self.n_freqs
        elif self.dependence == "energy":
            x_axis = self.ebounds
            x_axis_label = "Energy (keV)"
            #same issue as above for the bins, this will break when we
            #mask out frequency bins in the middle 
            labels = np.round(self.freq_bounds,1)
            units = "Hz"
            spec_number = self.n_freqs    
            data_bound = self.n_chans

        #this returns on the un-masked channel
        model = self.eval_model(params=params)
        
        #if we're plotting data, also get the residuals
        if plot_data is True:
            #these residuals are stupid
            model_res,res_errors = self.get_residuals(self.model,residuals)
            if residuals == "delchi":
                reslabel = "$\\Delta\\chi$"
            else:
                reslabel = "Data/model"

        if self.units != "lags":
            if plot_data is True:
                #plot real and imaginary, or phase and modulus, including
                #model, data and residuals
                fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12.,6.),
                                                          sharex=True,
                                                          gridspec_kw={'height_ratios': [2, 1]})      
                ax2.hlines(0,x_axis[0],x_axis[-1],color='black',ls=':',zorder=3)

                for i in range(spec_number):
                    col="C"+str(i)
                    ax1.errorbar(x_axis,self.data[i*data_bound:(i+1)*data_bound],
                                 yerr=self.data_err[i*data_bound:(i+1)*data_bound],
                                 marker='o',color=col,linestyle='',
                                 label=f"{labels[i]}-{labels[i+1]} {units}")
                    ax2.errorbar(x_axis,self.data[self.n_bins+i*data_bound:self.n_bins+(i+1)*data_bound],
                                 yerr=self.data_err[self.n_bins+i*data_bound:self.n_bins+(i+1)*data_bound],
                                 marker='o',color=col,linestyle='',
                                 label=f"{labels[i]}-{labels[i+1]} {units}")
                    ax3.errorbar(x_axis,model_res[i*data_bound:(i+1)*data_bound],
                                 yerr=res_errors[i*data_bound:(i+1)*data_bound],
                                 marker='o',linestyle='',color=col,zorder=2)
                    ax4.errorbar(x_axis,model_res[self.n_bins+i*data_bound:self.n_bins+(i+1)*data_bound],
                                 yerr=res_errors[self.n_bins+i*data_bound:self.n_bins+(i+1)*data_bound],
                                 marker='o',linestyle='',color=col, zorder=2)
                    ax1.plot(x_axis,model[i*data_bound:(i+1)*data_bound],linewidth=3,zorder=3)
                    ax2.plot(x_axis,model[self.n_bins+i*data_bound:self.n_bins+(i+1)*data_bound],
                             linewidth=3,zorder=3)
                ax1.set_xscale("log")
                ax1.set_yscale("log")  
                ax2.legend(loc="best",ncol=2)
                ax2.set_xscale("log")
                ax3.set_xscale("log")
                ax3.set_xlabel(x_axis_label) 
                ax3.set_ylabel(reslabel)  
                ax4.set_xlabel(x_axis_label) 
                ax4.set_ylabel(reslabel)
                if self.units == "polar":
                    ax1.set_ylabel("Modulus")
                    ax2.set_ylabel("Phase")
                else:
                    ax1.set_ylabel("Real part")
                    ax2.set_ylabel("Imaginary part")
                if residuals == "delchi":
                    ax3.hlines(0,x_axis[0],x_axis[-1],color='black',ls=':',zorder=4)
                    ax4.hlines(0,x_axis[0],x_axis[-1],color='black',ls=':',zorder=4)
                elif residuals == "ratio":
                    ax3.hlines(1,x_axis[0],x_axis[-1],color='black',ls=':',zorder=4)
                    ax4.hlines(1,x_axis[0],x_axis[-1],color='black',ls=':',zorder=4)
            else:
                #plot real and imaginary, or phase and modulus,
                #including only the model
                fig, ((ax1),(ax2)) = plt.subplots(1,2,figsize=(12.,5.))  
                for i in range(spec_number):
                    col="C"+str(i)
                    ax1.plot(x_axis,model[i*data_bound:(i+1)*data_bound],
                             linewidth=3,color=col,zorder=3)
                    ax2.plot(x_axis,model[self.n_bins+i*data_bound:self.n_bins+(i+1)*data_bound],
                             linewidth=3,color=col,zorder=3,
                             label=f"{labels[i]}-{labels[i+1]} {units}") 
                ax2.hlines(0,x_axis[0],x_axis[-1],color='black',ls=':',zorder=3)
                ax2.legend(loc="best",ncol=2)
                ax1.set_xscale("log")
                ax2.set_xscale("log")
                ax1.set_xlabel(x_axis_label) 
                ax2.set_xlabel(x_axis_label) 
                if self.units == "polar":
                    ax1.set_ylabel("Modulus")
                    ax2.set_ylabel("Phase")
                else:
                    ax1.set_ylabel("Real part")
                    ax2.set_ylabel("Imaginary part")
        else:
            if plot_data is True:  
                #plot the lags, including model, data and residuals
                fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6.,6.),sharex=True,
                                              gridspec_kw={'height_ratios': [2, 1]})
                ax1.hlines(0,x_axis[0],x_axis[-1],color='black',ls=':',zorder=4)
                for i in range(spec_number):
                    col="C"+str(i)
                    ax1.errorbar(x_axis,self.data[i*data_bound:(i+1)*data_bound],
                                 yerr=self.data_err[i*data_bound:(i+1)*data_bound],
                                 marker='o',linestyle='',color=col,zorder=2,
                                 label=f"{labels[i]}-{labels[i+1]} {units}")           
                    ax1.plot(x_axis,model[i*data_bound:(i+1)*data_bound],
                             linewidth=3,color=col,zorder=3) 
                    ax2.errorbar(x_axis,model_res[i*data_bound:(i+1)*data_bound],
                                 yerr=res_errors[i*data_bound:(i+1)*data_bound],
                                 linestyle='',marker='o',color=col,zorder=2)
                if residuals == "delchi": 
                    ax2.hlines(0,x_axis[0],x_axis[-1],color='black',ls=':',zorder=4)
                elif residuals == "ratio":
                    ax2.hlines(1,x_axis[0],x_axis[-1],color='black',ls=':',zorder=4)
                ax1.set_ylabel("Lag (s)")
                ax1.set_xscale("log")
                ax2.set_ylabel(reslabel)
                ax1.legend(loc="best",ncol=2)
                ax2.set_xlabel(x_axis_label)   
                ax2.set_xscale("log")
            else:
                #plot the lags, including only the model
                fig, ((ax1)) = plt.subplots(1,1,figsize=(6.,4.5)) 
                ax1.hlines(0,x_axis[0],x_axis[-1],color='black',ls=':',zorder=4)
                for i in range(spec_number):
                    col="C"+str(i)
                    ax1.plot(x_axis,model[i*data_bound:(i+1)*data_bound],
                             linestyle='',marker='o',color=col,
                             label=f"{labels[i]}-{labels[i+1]} {units}") 
                ax1.set_xscale("log")
                ax1.legend(loc="best",ncol=2)
                ax1.set_ylabel("Lag (s)")
                ax1.set_xlabel(x_axis_label)                
        
        fig.tight_layout()
        if return_plot is True:
            return fig
        else:
            return 

    def plot_model_2d(self,params=None,use_phase=False,residuals="delchi",return_plot=False):
    
        if self.dependence == "frequency":
            x_axis = self.freqs
            y_axis = self._ebounds_unmasked
            #here we need to look at the edges of each bin, NOT at the center
            #which is contained in ebounds
            channel_bounds = np.append(self.response.emin,self.response.emax[-1])
            labels = np.round(channel_bounds,1)
            units = "keV"
            spec_number = self.n_chans
            data_bound = self.n_freqs
        elif self.dependence == "energy":
            x_axis =  0.5*(self.freq_bounds[1:]+self.freq_bounds[:-1])
            y_axis = self._ebounds_unmasked
            labels = np.round(self.freq_bounds,1)
            spec_number = self.n_freqs    
            data_bound = self.n_chans 

        model = self.eval_model(params=params,mask=False)
        model_res,_ = self.get_residuals(model,residuals,use_masked=False)

        if self.units != "lags":
            if self.units == "polar":
                left_title = "Modulus"
                mid_title = "Phase"
            else:
                left_title = "Real"
                mid_title = "Imaginary" 

            '''
               elif self.dependence == "frequency":
                mask_twod = np.transpose(np.tile(self.ebounds_mask,self.n_freqs))
                plot_data = np.transpose(self._data_unmasked.reshape((self._all_chans,self.n_freqs)))
                plot_model = np.transpose(model.reshape((self._all_chans,self.n_freqs)))
                plot_res = np.transpose(model_res.reshape((self._all_chans,self.n_freqs)))
            mask_twod = mask_twod.reshape((self.n_freqs,self._all_chans))
            mask_twod = np.logical_not(mask_twod)          
            '''
            
            if self.dependence == "energy":
                mask_twod = np.tile(self.ebounds_mask,self.n_freqs)
                data_reformat = self._data_unmasked[:self._all_bins].reshape((self.n_freqs,self._all_chans))
                model_reformat = model[:self._all_bins].reshape((self.n_freqs,self._all_chans))
            elif self.dependence == "frequency":
                mask_twod = np.transpose(np.tile(self.ebounds_mask,self.n_freqs))
                data_reformat = np.transpose(self._data_unmasked[:self._all_bins].reshape((self._all_chans,self.n_freqs)))
                model_reformat =  np.transpose(model[:self._all_bins].reshape((self._all_chans,self.n_freqs)))
            mask_twod = mask_twod.reshape((self.n_freqs,self._all_chans))
            mask_twod = np.logical_not(mask_twod) 
            data_reformat = np.transpose(np.ma.masked_where(mask_twod, data_reformat))
            model_reformat = np.transpose(np.ma.masked_where(mask_twod, model_reformat))
            plot_info = [data_reformat,model_reformat]

            scale_min = np.min(self.data[:self.n_bins])
            scale_max = np.max(self.data[:self.n_bins])
    
            fig, axs = plt.subplots(2, 3, figsize=(15.,6.), sharex=True) 
            for row in range(2):
                ax = axs[row][0]
                left_plot = ax.pcolormesh(x_axis,y_axis,plot_info[row],cmap="viridis",
                                          shading='auto',rasterized=True,vmin=scale_min,vmax=scale_max)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylabel("Energy (keV)")
                ax.set_ylim([self.ebounds[0]-0.5*self.ewidths[0],
                             self.ebounds[-1]+0.5*self.ewidths[-1]])
            axs[0][0].set_title(left_title)
            ax.set_xlabel("Frequency (Hz)")            
            fig.subplots_adjust(wspace=0.075)
            cbar = fig.colorbar(left_plot, ax=axs[0:2,0],aspect = 40)
            cbar.formatter.set_powerlimits((0, 0))
            
            if self.dependence == "energy":
                data_reformat = self._data_unmasked[self._all_bins:].reshape((self.n_freqs,self._all_chans))
                model_reformat = model[self._all_bins:].reshape((self.n_freqs,self._all_chans))
            elif self.dependence == "frequency":
                data_reformat = np.transpose(self._data_unmasked[self._all_bins:].reshape((self._all_chans,self.n_freqs)))
                model_reformat =  np.transpose(model[self._all_bins:].reshape((self._all_chans,self.n_freqs)))
            data_reformat = np.transpose(np.ma.masked_where(mask_twod, data_reformat))
            model_reformat = np.transpose(np.ma.masked_where(mask_twod, model_reformat))
            plot_info = [data_reformat,model_reformat]
            
            scale_min = np.min(self.data[self.n_bins:])
            scale_max = np.max(self.data[self.n_bins:])
    
            for row in range(2):
                ax = axs[row][1]
                mid_plot = ax.pcolormesh(x_axis,y_axis,plot_info[row],cmap="cividis",
                                            shading='auto',rasterized=True,vmin=scale_min,vmax=scale_max)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_yticklabels([])
                ax.set_ylim([self.ebounds[0]-0.5*self.ewidths[0],
                             self.ebounds[-1]+0.5*self.ewidths[-1]])            
            axs[0][1].set_title(mid_title)
            ax.set_xlabel("Frequency (Hz)")
            cbar = fig.colorbar(mid_plot, ax=axs[0:2,1],aspect = 40)
            cbar.formatter.set_powerlimits((0, 0))

            if self.dependence == "energy":
                top_res = model_res[self._all_bins:].reshape((self._all_chans,self.n_freqs))
                bot_res = model_res[:self._all_bins].reshape((self._all_chans,self.n_freqs))
            elif self.dependence == "frequency":
                top_res = np.transpose(model_res[self._all_bins:].reshape((self._all_chans,self.n_freqs)))
                bot_res = np.transpose(model_res[:self._all_bins].reshape((self._all_chans,self.n_freqs)))
            top_res = np.transpose(np.ma.masked_where(mask_twod, top_res))
            bot_res = np.transpose(np.ma.masked_where(mask_twod, bot_res))
            plot_info = [top_res,bot_res]

            for row in range(2):
                ax = axs[row][2]
                filtered_row = self._filter_2d_by_mask(np.array(plot_info[row]),self.ebounds_mask)
                res_min = np.min([np.min(filtered_row),-1])
                res_max = np.max([np.max(filtered_row),1])
                
                #res_min = np.min(np.append(plot_info[row].reshape(self._all_bins),-0.1))
                #res_max = np.max(np.append(plot_info[row].reshape(self._all_bins),0.1))
                res_norm = TwoSlopeNorm(vmin=res_min,vcenter=0,vmax=res_max) 
                mid_plot = ax.pcolormesh(x_axis,y_axis,plot_info[row],cmap="BrBG",
                                            shading='auto',rasterized=True,norm=res_norm)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_yticklabels([])
                ax.set_ylim([self.ebounds[0]-0.5*self.ewidths[0],
                             self.ebounds[-1]+0.5*self.ewidths[-1]])
                cbar = fig.colorbar(mid_plot, ax=ax)
                cbar.formatter.set_powerlimits((0, 0))
            axs[0][2].set_title(mid_title+" residuals")
            axs[1][2].set_title(left_title+" residuals")
            ax.set_xlabel("Frequency (Hz)")
        else:
            fig, ((ax1),(ax2),(ax3)) = plt.subplots(1, 3, figsize=(15.,4.), sharex=True)             
            if self.dependence == "energy":               
                mask_twod = np.tile(self.ebounds_mask,self.n_freqs)
                plot_data = self._data_unmasked.reshape((self.n_freqs,self._all_chans))
                plot_model = model.reshape((self.n_freqs,self._all_chans))
                plot_res = model_res.reshape((self.n_freqs,self._all_chans))
            elif self.dependence == "frequency":
                mask_twod = np.transpose(np.tile(self.ebounds_mask,self.n_freqs))
                plot_data = np.transpose(self._data_unmasked.reshape((self._all_chans,self.n_freqs)))
                plot_model = np.transpose(model.reshape((self._all_chans,self.n_freqs)))
                plot_res = np.transpose(model_res.reshape((self._all_chans,self.n_freqs)))
            mask_twod = mask_twod.reshape((self.n_freqs,self._all_chans))
            mask_twod = np.logical_not(mask_twod) 
            if use_phase is True:
                plot_data = plot_data*(2.*np.pi*x_axis.reshape(self.n_freqs,1))
            plot_data = np.transpose(np.ma.masked_where(mask_twod, plot_data))
            color_min = np.min([np.min(plot_data),-0.01])
            color_max = np.max([np.max(plot_data),0.01])
            lag_norm = TwoSlopeNorm(vmin=color_min,vcenter=0,vmax=color_max) 
            data_plot = ax1.pcolormesh(x_axis,y_axis,plot_data,cmap="BrBG",
                                        shading='auto',linewidth=0,norm=lag_norm)
            
            ax1.set_title("Data")                
            fig.colorbar(data_plot, ax=ax1)
            if use_phase is True:
                plot_model = plot_model*(2.*np.pi*x_axis.reshape(self.n_freqs,1))
            plot_model = np.transpose(np.ma.masked_where(mask_twod, plot_model))
            lag_norm = TwoSlopeNorm(vmin=color_min,vcenter=0,vmax=color_max) 
            model_plot = ax2.pcolormesh(x_axis,y_axis,plot_model,cmap="BrBG",
                                        shading='auto',linewidth=0,norm=lag_norm)
            ax2.set_title("Model")                
            fig.colorbar(model_plot, ax=ax2)    
            
            plot_res = np.transpose(np.ma.masked_where(mask_twod, plot_res))
            res_min = np.min([np.min(plot_res),-1])
            res_max = np.max([np.max(plot_res),1])
            res_norm = TwoSlopeNorm(vmin=res_min,vcenter=0,vmax=res_max) 
            res_plot = ax3.pcolormesh(x_axis,y_axis,plot_res,cmap="BrBG",
                                        shading='auto',linewidth=0,norm=res_norm)
    
            ax3.set_title("Residuals")                
            fig.colorbar(res_plot, ax=ax3)
            
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_ylim([self.ebounds[0]-0.5*self.ewidths[0],self.ebounds[-1]+0.5*self.ewidths[-1]])
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Energy (keV)")
            
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_ylim([self.ebounds[0]-0.5*self.ewidths[0],1.05*self.ebounds[-1]+0.5*self.ewidths[-1]])
            ax2.set_yticklabels([])
            ax2.set_xlabel("Frequency (Hz)")
            
            ax3.set_xscale("log")
            ax3.set_yscale("log")
            ax3.set_ylim([self.ebounds[0]-0.5*self.ewidths[0],self.ebounds[-1]+0.5*self.ewidths[-1]])
            ax3.set_yticklabels([])
            ax3.set_xlabel("Frequency (Hz)")
        
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
    from astropy.io.fits.card import Undefined, UNDEFINED
    
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

def load_lc(path):
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
        #convert from astropy to numpy - this is annoying
        #for 2d arrays hence the horror below
        gti = np.zeros((len(gti_data),2))
        for i in range(len(gti_data)):
            gti[i][0] = gti_data[i][0]-gti_data[0][0]
            gti[i][1] = gti_data[i][1]-gti_data[0][0]

    return time_bins, counts, gti
