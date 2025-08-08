import numpy as np
from lmfit import fit_report, minimize
from lmfit import Parameters
from SimpleFit import SimpleFit

class JointFit():
    """
    Generic joint fitting class. Use this class if you have multiple 
    non-simultaneous observations that you wish to share parameters 
    between or mutliple simultaneous observations that share a single model.
    
    Attributes:
    ------------
    joint : dict{Fit... objects and/or list(Fit... objects)}
        Dictionary containing named Fit... objects to be joint fitted. By
        default, observations that share parameters completely (simultaneous or
        different data products of observations) are packaged together in lists.
        
    fit_result: lmfit.MinimizeResult
        A lmfit MinimizeResult, which stores the result (including best-fitting 
        parameter values, fit statistics etc) of a fit after it has been run. 
    """
    
    def __init__(self):
        self.joint = {}
        self.joint_params = {}
        self.fit_result = None
        
    def add_fitobj(self,fitobj,name):
        """
        Adds a model or models to the joint fitting hierarchy. If a list of 
        fitting objects is added, it is assumed that the objects are intended
        as a simultaneous fit.

        Parameters
        ----------
        fitobj : Fit... object or list of Fit... objects
            the Fit... object of the observation and model. In the case of
            multiple fit objects, they must all share the same underlying model
            as all of their parameters will be linked.
        
        name: str
            name of the model
        """
        if type(fitobj) == list:
            for obj in fitobj:
                if issubclass(obj,SimpleFit):
                    pass
                else:
                    raise TypeError("Invalid object passed")
        else:
            if issubclass(fitobj,SimpleFit):
                pass
            else:
                raise TypeError("Invalid object passed")
        #if simultaneous observations (so same underlying model)
        if type(fitobj) == list: 
            self.joint[name] = fitobj
            if len(fitobj) > 1: #check if actually multiple models
                first_fitobj = fitobj[0]
                #links all model parameters to first model in list
                for i in range(1,len(fitobj)): 
                    self.link_params(first_fitobj, fitobj[i])
                #pulls parameters names and saves to dictionary for model
                #evaluation later
                params = []
                for key in first_fitobj.model_params.valuesdict().keys():
                    #iterates through current fit objects
                    for joint_obs in self.joint_params:
                        #checks if parameter matches any previous model 
                        #parameters
                        if key in self.joint_params[joint_obs]:
                            print(f"""
                                  Caution: {key} is already a model parameter.
                                  Do you intend for these parameters to be linked?
                                  If not, give it a different name to differentiate
                                  between multiple instances of the same type for
                                  different models.
                                  """)
                    params.append(key)
                self.joint_params[name] = params
            else:
                raise TypeError("""
                                Unnecessary list of models due to single entry,
                                either add other simultaneous observations
                                or only add the model as a single entry.
                                """)
        else: #single observation case
            self.joint[name] = fitobj
            #pulls parameters names and saves to dictionary for model
            #evaluation later
            params = []
            for key in fitobj.model_params.valuesdict().keys():
                for joint_obs in self.joint_params:
                    if key in self.joint_params[joint_obs]:
                        print(f"""
                              Caution: {key} is already a model parameter.
                              Do you intend for these parameters to be linked?
                              If not, give it a different name to differentiate
                              between multiple instances of the same type for
                              different models.
                              """)
                params.append(key)
            self.joint_params[name] = params
            
    def link_params(self,first_fitobj,second_fitobj,param_names=None):
        """
        Links parameters between models.

        Parameters
        ----------
        first_fitobj : Fit... object 
            DESCRIPTION
        second_fitobj : Fit... object 
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
            if (((getattr(first_fitobj.model, '__module__', None) != "lmfit.compositemodel")&
                (getattr(first_fitobj.model, '__module__', None) != "lmfit.model"))|
                ((getattr(second_fitobj.model, '__module__', None) != "lmfit.compositemodel")&
                    (getattr(second_fitobj.model, '__module__', None) != "lmfit.model"))):  
                raise AttributeError("The model input must be an LMFit Model or CompositeModel object")
            #makes class share single lmfit.Parameters object
            second_fitobj.model_params = first_fitobj.model_params
        
        if type(param_names) == list:
            #check all parameters are shared
            if ((set(param_names) <= first_fitobj.param_names)&
                (set(param_names) <= second_fitobj.param_names)):
                pass
            else: #if parameters are not shared, soft error
                print("Not all parameters inputted are shared between models")
                return
            
            #forces models to share parameter objects
            for name in param_names:
                second_fitobj.model_params[name] = first_fitobj.model_params[name]
            print("Multiple parameters linked")
            
        elif type(param_names) == str:
            #check parameter is shared
            if ((param_names in first_fitobj.param_names)&
                (param_names in second_fitobj.param_names)):
                pass
            else: #if parameters are not shared, soft error
                print(f"{param_names} is not shared between the models")
                return
            
            print("Single parameter linked")
        else:
            raise TypeError("Input parameter name or list of parameter names")
    
    def eval_model(self,params,names = None):
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
            names = self.joint.keys()
        
        #creates structure to return model results
        model_hierarchy = {}
        
        for name in names:
            if name not in self.joint.keys():
                raise AttributeError(f"{name} is not in model hierarchy")
            #retrieves model or models based on dictionary name
            fitobjs = self.joint[name] 
            model_param_names = self.joint_params[name]
            model_params = Parameters()
            for key in model_param_names:
                model_params.add_many(params[key])
            if type(fitobjs) == list: #if simultaneous, evaluate each one.
                model_results = []
                for fit_obj in fitobjs:
                    model_results.append(fit_obj.eval_model(model_params))
            else:
                model_results = fitobjs.eval_model(model_params)
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
        if self.joint == {}:
            raise AttributeError("No loaded observations or models")
            
        if names == None: #retrieves all models
            names = self.joint.keys()
        elif type(names) == str:
            names = [names]
            
        if type(names) != list:
            raise TypeError("Inputted names are not valid type")
        else:
            residual_dict = self.eval_model(params,names)
            residuals = []
            for name in names:
                obs_residuals = residual_dict[name]
                residuals.append(np.asarray(obs_residuals).flatten())
            residuals = np.asarray(residuals).flatten()
        return residuals
    
    def fit_data(self,algorithm='leastsq',names=None):
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
        if names == None:
            names = self.joint.keys()
        
        #retrieves all relevant parameters from models being fitted
        self.model_params = Parameters()
        for name in names:
            #collects all parameter names of observation
            model_param_names = self.joint_params[name]
            #adds all parameters to overarching joint fit (parameters with same
            #name are assumed to be linked and are overwritten).
            for key in model_param_names:
                self.model_params.add_many(self.joint[name].model_params[key])
            
        self.fit_result = minimize(self._minimizer,self.model_params,
                                   method=algorithm,args=[names])
        print(fit_report(self.fit_result,show_correl=False))
        fit_params = self.fit_result.params
        self.set_params(fit_params)
        return
    
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
            names = self.joint.keys()
        
        if type(names) == list:
            for name in names:
                print(f"{name}: \n")
                print("-----------------------")
                self.joint[name].print_model()
                print("-----------------------")
        else:
            print(f"{names}: \n")
            print("-----------------------")
            self.joint[names].print_model()
            print("-----------------------")
        
    def print_fit_results(self):
        """
        This method prints the current fit results.
        """
        if self.fit_result != None:
            print(fit_report(self.fit_result,show_correl=False))
        else:
            print("No current fit result.")