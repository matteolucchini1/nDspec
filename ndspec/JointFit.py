from lmfit import fit_report, minimize

class JointFit():
    """
    Generic joint fitting class. Use this class if you have multiple 
    non-simultaneous observations that you wish to share parameters 
    between or mutliple simultaneous observations that share a single model.
    
    Attributes:
    ------------
    hierarchy : dict{Fit... objects and/or list(Fit... objects)}
        Dictionary containing named Fit... objects to be joint fitted. By
        default, observations that share parameters completely (simultaneous or
        different data products of observations) are packaged together in lists.
        
    fit_result: lmfit.MinimizeResult
        A lmfit MinimizeResult, which stores the result (including best-fitting 
        parameter values, fit statistics etc) of a fit after it has been run. 
    """
    
    def __init__(self):
        self.hierarchy = {}
        self.fit_result = None
        
    def add_model(self,model,name):
        """
        Adds a model or models to the joint fitting hierarchy. If a list of 
        fitting objects is added, it is assumed that the objects are intended
        as a simultaneous fit.

        Parameters
        ----------
        model : Fit... object or list of Fit... objects
            the Fit... object of the observation and model. In the case of
            multiple fit objects, they must all share the same underlying model
            as all of their parameters will be linked.
        
        name: str
            name of the model
        """
        
        self.hierarchy[name] = model
        
        #if simultaneous observations (so same underlying model)
        if type(model) == list: 
            if len(model) > 1: #check if actually multiple models
                first_obs = model[0]
                #links all model parameters to first model in list
                for i in range(1,len(model)): 
                    self.link_params(first_obs, model[i])
            else:
                raise TypeError("""
                                Unnecessary list of models due to single entry,
                                either add other simultaneous observations
                                or only add the model as a single entry.
                                """)
    
    def link_params(self,first_obs,second_obs,param_names=None):
        """
        Links parameters between models.

        Parameters
        ----------
        first_obs : Fit... object 
            DESCRIPTION
        second_obs : Fit... object 
            DESCRIPTION
        param_names : str or list(str), optional
            Parameter names of parameters to link between models. The default 
            is to share all parameters together

        Returns
        -------
        None.

        """
        if param_names == None: #defaults to all parameters
            #checks that both models are correctly specified
            if (((getattr(first_obs.model, '__module__', None) != "lmfit.compositemodel")&
                (getattr(first_obs.model, '__module__', None) != "lmfit.model"))|
                ((getattr(second_obs.model, '__module__', None) != "lmfit.compositemodel")&
                    (getattr(second_obs.model, '__module__', None) != "lmfit.model"))):  
                raise AttributeError("The model input must be an LMFit Model or CompositeModel object")
            #makes class share single lmfit.Parameters object
            second_obs.model_params = first_obs.model_params
        
        if type(param_names) == list:
            #check all parameters are shared
            if ((set(param_names) <= first_obs.param_names)&
                (set(param_names) <= second_obs.param_names)):
                pass
            else: #if parameters are not shared, soft error
                print("Not all parameters inputted are shared between models")
                return
            
            #forces models to share parameter objects
            for name in param_names:
                second_obs.model_params[name] = first_obs.model_params[name]
            print("Multiple parameters linked")
            
        elif type(param_names) == str:
            #check parameter is shared
            if ((param_names in first_obs.param_names)&
                (param_names in second_obs.param_names)):
                pass
            else: #if parameters are not shared, soft error
                print(f"{param_names} is not shared between the models")
                return
            
            print("Single parameter linked")
        else:
            raise TypeError("Input parameter name or list of parameter names")
    
    def eval_model(self,names = None):
        """
        This method is used to evaluate and return the model values of models 
        in the hierarchy.
        
        Parameters:
        ------------
        names: list(str), default None
            names of the models that should be evalualated. Defaults to
            evaluating all models.
        
        Returns:
        --------
        model: np.array(float)
            All models are evaluated and returned as a dictionary, corresponding
            to the top-level hierarchy.
        """
        if names == None: #retrieves all models
            names = self.hierarchy.keys()
        
        #creates structure to return model results
        model_hierarchy = {}
        
        for name in names:
            if name not in self.hierarchy.keys():
                raise AttributeError(f"{name} is not in model hierarchy")
            #retrieves model or models based on dictionary name
            models = self.hierarchy[name] 
            if type(models) == list: #if simultaneous, evaluate each one.
                model_results = []
                for model in models:
                    model_results.append(model.eval_model())
            else:
                model_results = models.eval_model()
            model_hierarchy[name] = model_results
        
        return model_hierarchy
        
    def _minimizer(self,params,names = None):
        """
        This method is used exclusively when running a minimization algorithm.
        It evaluates the models for an input set of parameters, and then returns 
        the residuals in units of contribution to the total chi squared 
        statistic.
        
        Parameters:
        -----------                         
        params: lmfit.Parameters or list[lmfit.Parameters]
            The parameter values to use in evaluating the model. These will vary 
            as the fit runs.
        
        names: list(str), default None
            names of the models that should be evalualated. Defaults to
            evaluating all models.
            
        Returns:
        --------
        residuals: np.array(float)
            An array of the same size as the data, containing the model 
            residuals in each bin.            
        """
        if names == None: #retrieves all models
            names = self.hierarchy.keys()
        
        if type(params) == list:
            total_residuals = []
            for pars,name in zip(params,names):
                total_residuals.append(self.hierarchy[name]._minimizer(params))
        else:
            if self.likelihood is None:
                model = self.model.eval(params,freq=self.freqs)
                residuals = (self.data-model)/self.data_err
            else:
                raise AttributeError("custom likelihood not implemented yet")
            
        return residuals
    
    def print_models(self,names=None):
        """
        Prints the models contained within the joint fit. Defaults to printing
        all models, but users can filter models using the names parameter.

        Parameters
        ----------
        names : str or list(str), optional
            names of the models to be printed. The default is to print all 
            models.
        """
        if names == None:
            names = self.hierarchy.keys()
        
        if type(names) == list:
            for name in names:
                print(f"{name}: \n")
                print("-----------------------")
                self.hierarchy[name].print_model()
                print("-----------------------")
        else:
            print(f"{names}: \n")
            print("-----------------------")
            self.hierarchy[names].print_model()
            print("-----------------------")
        
    def print_fit_results(self):
        """
        This method prints the current fit results.
        """
        if self.fit_result != None:
            print(fit_report(self.fit_result,show_correl=False))
        else:
            print("No current fit result.")