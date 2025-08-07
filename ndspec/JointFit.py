
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
        model : Fit... object or list of objects
            the Fit... object of the observation and model. In the case of
            multiple fit objects, they must all share the same underlying model
            as all of their parameters will be linked.
        
        name: str
            name of the model
        """
        
        self.hierarchy[name] = model
    
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
        
        