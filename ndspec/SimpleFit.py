import numpy as np

from lmfit import fit_report, minimize 

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
        
    _data_unmasked, _data_err_unmasked: np.array(float)
        The arrays of every data bin and its error, regardless of which ones are
        ignored or noticed during the fit. Used exclusively to enable book 
        keeping internal to the fitter class.          
    """ 

    def __init__(self):
        self.model = None
        self.model_params = None
        self.likelihood = None
        self.fit_result = None
        self.data = None
        self.data_err = None
    pass

    def _set_unmasked_data(self):
        """
        This initializer method is used to set up the unmasked arrays for later 
        book-keeping. Depending on the dependence of the fit, it initializes 
        different internal unmasked arrays.        
        """

        self._data_unmasked = self.data
        self._data_err_unmasked = self.data_err

        if isinstance(self,EnergyDependentFit) is True:
            self._emin_unmasked = self.response.emin
            self._emax_unmasked = self.response.emax
            self._ebounds_unmasked = self.ebounds
            self._ewidths_unmasked = self.ewidths
            self._all_chans = self._ebounds_unmasked.size
            self.n_chans = self._all_chans
            #additional internal arrays if we're doing spectral timing
            if isinstance(self,FrequencyDependentFit) is True:
                self._all_bins = self._all_freqs*self._all_chans
                self.n_bins = self._all_bins                
            #future: add if spectral polarimetry
        elif isinstance(self,FrequencyDependentFit) is True:
            #note: these assignements are redundant for just a PSD fit 
            self._freqs_unmasked = self.freqs           
            self.n_freqs = self.freqs.size
            self._all_freqs = self.n_freqs
            #future: add if timing polarimetry within the fit 
        return

    def _filter_2d_by_mask(self,array):
        """
        This method is used to filter two-dimensional data (for example, a cross
        spectrum) after users define a range of energy channels or Fourier 
        frequency bins to ignore. 
        
        Parametrers:
        ------------
        array: np.float 
            The one-dimensional array containing the (flattened) two-dimensional 
            data or model to be filtered 
            
        Output:
        -------
        filter_arr: np.float 
            The one-dimensional array filtered by the two-d mask defind by the 
            noticed frequency bins and channels.
        """
        
        self.n_bins = self.n_chans*self.n_freqs
        
        if self.dependence == "energy":
            twod_mask = self.freqs_mask.reshape((self._all_freqs,1))* \
                        self.ebounds_mask.reshape((1,self._all_chans))
        elif self.dependence == "frequency":
            twod_mask = self.ebounds_mask.reshape((self._all_chans,1))* \
                        self.freqs_mask.reshape((1,self._all_freqs))  
        else:
            raise AttributeError("Data dependence not specified")
        twod_mask = np.array(twod_mask).flatten()

        if self.units != "lags":
            filter_first_dim = np.extract(twod_mask,array[:self._all_bins])
            #error_filter_first_dim = np.extract(twod_mask,array[:self._all_bins])
            filter_second_dim = np.extract(twod_mask,array[self._all_bins:],)
            #error_filter_second_dim = np.extract(twod_mask,array[self._all_bins:])
            filter_arr = np.append(filter_first_dim,filter_second_dim)
            #self.data_err = np.append(error_filter_first_dim,error_filter_second_dim)              
        else:
            filter_arr = np.extract(twod_mask,array)
            #self.data_err = np.extract(twod_mask,array)
        return filter_arr
    
    def set_model(self,model,params=None):
        """
        This method is used to pass the model users want to fit to the data. 
        Optionally it is also possible to pass the initial parameter values of 
        the model. 
        
        Parameters:
        -----------            
        model: lmfit.model or lmfit.compositemodel 
            The lmfit wrapper of the model one wants to fit to the data. 
            
        params: lmfit.Parameters, default: None 
            The parameter values from which to start evalauting the model during
            the fit. If it is not provided, all model parameters will default 
            to 0, set to be free, and have no minimum or maximum bound. 
        """

        if ((getattr(model, '__module__', None) != "lmfit.compositemodel")&
            (getattr(model, '__module__', None) != "lmfit.model")):  
        #if isinstance(model,lmfit.CompositeModel) is False:
            raise AttributeError("The model input must be an LMFit Model or CompositeModel object")
        
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
        params: lmfit.parameter
            The parameter values from which to start evalauting the model during
            the fit.  
        """
        
        #maybe find a way to go through the parameters of the model, and make sure 
        #the object passed contains the same parameters?
        if getattr(params, '__module__', None) != "lmfit.parameter":  
#        if isinstance(params,lmfit.Parameters) is False:
            raise AttributeError("The parameters input must be an LMFit Parameters object")
        
        self.model_params = params
        return 

    def get_residuals(self,res_type,model=None,use_masked=True):    
        """
        This methods return the residuals (either as data/model, or as 
        contribution to the total chi squared) of the input model, given the 
        parameters set in model_parameters, with respect to the data. 
        
        Parameters:
        -----------
        res_type: string 
            If set to "ratio", the method returns the residuals defined as 
            data/model. If set to "delchi", it returns the contribution of 
            each energy channel to the total chi squared.

        use_masked: bool, default True 
            A flag to decide whether to compare the model against the masked or 
            unmasked data. 
            
        Returns:
        --------
        residuals: np.array(float)
            An array of the same size as the data, containing the model 
            residuals in each channel.
            
        bars: np.array(float)
            An array of the same size as the residuals, containing the one sigma 
            range for each contribution to the residuals.           
        """

        if model is None:
            model = self.eval_model()
        
        if use_masked is True:
            data = self.data
            error = self.data_err
        elif use_masked is False:
            data = self._data_unmasked
            error = self._data_err_unmasked

        if res_type == "ratio":
            residuals = data/model
            bars = error/model
        elif res_type == "delchi":
            residuals = (data-model)/error
            bars = np.ones(len(data))
        else:
            raise ValueError("The only supported residual types are ratio and delta chi")
            
        return residuals, bars

    def print_fit_stat(self):
        """
        This method compares the model defined by the user, using the last set
        of parameters to have been set in the class, to the data stored. It then
        prints the chi-squared goodness-of-fit to terminal, along with the 
        number of data bins, free parameters and degrees of freedom. 
        """
        
        if self.likelihood is None:
            res, err = self.get_residuals("delchi")
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
    """
    #changing response here for the sake of testing stuff
    def __init__(self):   
        self.energs = 0.5*(self.response.energ_hi+self.response.energ_lo)
        self.energ_bounds = self.response.energ_hi-self.response.energ_lo
        self.ebounds = 0.5*(self.response.emax+self.response.emin)
        self.ewidths = self.response.emax - self.response.emin
        self.ebounds_mask = np.full((self.response.n_chans), True)
        pass
       
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

        #filter 2d data is more complex so it is moved to its own method for 
        #simplicity
        if isinstance(self,FrequencyDependentFit) is True:
            self.data = self._filter_2d_by_mask(self._data_unmasked)
            self.data_err = self._filter_2d_by_mask(self._data_err_unmasked)
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

        #filter 2d data is more complex so it is moved to its own method for 
        #simplicity
        if isinstance(self,FrequencyDependentFit) is True:
            self.data = self._filter_2d_by_mask(self._data_unmasked)
            self.data_err = self._filter_2d_by_mask(self._data_err_unmasked)
        else:
            self.data = np.extract(self.ebounds_mask,self._data_unmasked)
            self.data_err = np.extract(self.ebounds_mask,self._data_err_unmasked)              
        return

class FrequencyDependentFit():
    """
    Internal book-keeping class used to manage noticing or ignoring Fourier  
    frequency bins. 
    
    Stores the full (unmasked) Fourier bins, and data arrays, a mask
    used to track which bins/data points are noticed or ignored.

    Attributes:
    -----------    
    _freqs_unmasked: np.array(float)
        If the data and model explicitely depend on Fourier frequency (e.g. a
        power spectrum), this is the array of Fourier frequency over which all 
        data and model are defined, including bins that are ignored in the fit. 
        
        If instead the data depends from some other energy (e.g. energy), it 
        contains both noticed and ignored frequency intervals over which to 
        produce spectral-timing products. For example, a user might input a set 
        of 7 ranges of frequencies to calculate lag energy spectra, but only 
        want to consider the first and last 3, and ignore the middle one.
    
    freqs_mask np.array(bool)
        The array of Fourier frequencies that are either ignored or noticed 
        during the fit. A given channel i is noticed if freqs_mask[i] is True,
        and ignored if it is false.      
    
    n_freqs: int 
        The number of Fourier frequency bins that are noticed in the fit. 
    
    n_bins: int 
        Only used for two-dimensional data fitting. Defined as the number of 
        noticed channels, times the number of bins in the second dimension 
        (e.g. Fourier frequency).
        
    _all_bins: int 
        Only used for two-dimensional data fitting. Defined as the total number 
        of  channels, times the number of bins in the second dimension 
        (e.g. Fourier frequency).   
    """

    def __init__(self,freqs):
        self._freqs_unmasked = freqs
        self.freqs = self._freqs_unmasked
        if self.dependence == "frequency":
            self._all_freqs = self._freqs_unmasked.size
        else:
            self._all_freqs = self._freqs_unmasked.size-1
        self.n_freqs = self._all_freqs
        self.freqs_mask = np.full((self._all_freqs), True)
        pass

    def ignore_frequencies(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        ignore selected frequencies based on user-supplied bounds bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored frequency interval.
        bound_hi : float
            Higher bound of ignored frequency interval.    
        """
        
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Frequency bounds must be floats or integers")
       
        if self.dependence == "frequency":
            #this is called for a regular frequency-dependent product
            self.freqs_mask = ((self._freqs_unmasked<bound_lo)|
                               (self._freqs_unmasked>bound_hi))&self.freqs_mask
            self.freqs = np.extract(self.freqs_mask,self._freqs_unmasked)
        else:
            #this is for products that do not depend on energy explicitely, but 
            #only implicitely - for example, lag-energy data.
            fmin = self._freqs_unmasked[:-1]
            fmax = self._freqs_unmasked[1:]
            self.freqs_mask = ((fmin<bound_lo)|
                               (fmax>bound_hi))&self.freqs_mask
            self.freq_bounds = np.extract(self.freqs_mask,self._freqs_unmasked)
        self.n_freqs = self.freqs_mask[self.freqs_mask==True].size

        #filter 2d data is more complex so it is moved to its own method for 
        #simplicity
        if isinstance(self,EnergyDependentFit) is True:
            self.data = self._filter_2d_by_mask(self._data_unmasked)
            self.data_err = self._filter_2d_by_mask(self._data_err_unmasked)
        else:
            self.data = np.extract(self.freqs_mask,self._data_unmasked)
            self.data_err = np.extract(self.freqs_mask,self._data_err_unmasked)              
        return

    def notice_frequencies(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        ignore selected frequencies based on user-supplied bounds bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored frequency interval.
        bound_hi : float
            Higher bound of ignored frequency interval.    
        """
        
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Frequency bounds must be floats or integers")

        if self.dependence == "frequency":
            self.freqs_mask = self.freqs_mask|np.logical_not(
                              (self._freqs_unmasked<bound_lo)|
                              (self._freqs_unmasked>bound_hi))
            self.freqs = np.extract(self.freqs_mask,self._freqs_unmasked)
        else:
            fmin = self._freqs_unmasked[:-1]
            fmax = self._freqs_unmasked[1:]
            self.freqs_mask = self.freqs_mask|np.logical_not(
                              (fmin<bound_lo)|
                              (fmax>bound_hi))
            self.freq_bounds = np.extract(self.freqs_mask,self._freqs_unmasked)
        self.n_freqs = self.freqs_mask[self.freqs_mask==True].size

        #filter 2d data is more complex so it is moved to its own method for 
        #simplicity
        if isinstance(self,EnergyDependentFit) is True:
            self.data = self._filter_2d_by_mask(self._data_unmasked)
            self.data_err = self._filter_2d_by_mask(self._data_err_unmasked)
        else:
            self.data = np.extract(self.freqs_mask,self._data_unmasked)
            self.data_err = np.extract(self.freqs_mask,self._data_err_unmasked)              
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
        hdr = spectrum["SPECTRUM"].header
        spectrum_data = spectrum['SPECTRUM'].data
        channels = spectrum_data['CHANNEL']
        counts = spectrum_data['COUNTS']
        #check that the spectrum and response have the same mission and channel 
        #number         
        mission_spectrum = hdr["TELESCOP"]
        instrument_spectrum = hdr["INSTRUME"]
        if mission_spectrum != response.mission:
            raise NameError("Observatory in the spectrum different from the response")
        if instrument_spectrum != response.instrument:
            raise NameError("Instrument in the spectrum different from the response")        
        
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
