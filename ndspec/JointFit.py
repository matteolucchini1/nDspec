
class JointFit():
    """
    Generic joint fitting class. Use this class if you have multiple 
    non-simultaneous observations that you wish to share parameters 
    between or mutliple simultaneous observations that share a single model.
    
    Attributes:
    ------------
    
    """
    
    def __init__(self):
        self.hierarchy = {}
        
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
                base_model = model[0]
                #links all model parameters to first model in list
                for i in range(1,len(model)): 
                    self.link_params(base_model, model[i])
            else:
                raise TypeError("""
                                Unnecessary list of models due to single entry,
                                either add other simultaneous observations
                                or 
                                """)
    
    def link_params(self,base_model,linked_model,param_names=None):
        """
        Links parameters between models.

        Parameters
        ----------
        base_model : Fit... object 
            DESCRIPTION
        linked_model : Fit... object 
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
            if (((getattr(base_model.model, '__module__', None) != "lmfit.compositemodel")&
                (getattr(base_model.model, '__module__', None) != "lmfit.model"))|
                ((getattr(base_model.model, '__module__', None) != "lmfit.compositemodel")&
                    (getattr(base_model.model, '__module__', None) != "lmfit.model"))):  
                raise AttributeError("The model input must be an LMFit Model or CompositeModel object")
            #makes class share single lmfit.Parameters object
            linked_model.model_params = base_model.model_params
        
        if type(param_names) == list:
            #check all parameters are shared
            if ((set(param_names) <= base_model.param_names)&
                (set(param_names) <= linked_model.param_names)):
                pass
            else: #if parameters are not shared, soft error
                print("Not all parameters inputted are shared between models")
                return
            
            #forces models to share parameter objects
            for name in param_names:
                linked_model.model_params[name] = base_model.model_params[name]
            print("Multiple parameters linked")
            
        elif type(param_names) == str:
            #check parameter is shared
            if ((param_names in base_model.param_names)&
                (param_names in linked_model.param_names)):
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