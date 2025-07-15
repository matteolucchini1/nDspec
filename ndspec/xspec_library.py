import ctypes as ct
import numpy as np
from functools import wraps
import os

#note: this is not compatible with Relxill specifically, will need a different wrapper urgh
class XspecLibrary:
    def __init__(self, lib_path=None,pars_path=None):
        # Default library path
        if lib_path is None:
            headas_path = os.environ.get("HEADAS")
            if headas_path:
                lib_path = headas_path + f"/../Xspec/{os.path.basename(headas_path)}/lib/libXSFunctions.so"
            else:
                raise EnvironmentError("HEADAS environment variable not set.")

        self._all_info = self.parse_models(pars_path)
        self.models_info = {}

        # Load the library
        self.lib = ct.cdll.LoadLibrary(lib_path)
           
    def initialize_heasoft(self):
        init_call = self.lib.fninit_
        init_call()
    
    #this parses all the models and their parameters; next up, isolate just the model that I want 
    def parse_models(self,input_file=None):
        if input_file is None:
            headas_path = os.environ.get("HEADAS")
            input_file =  headas_path + f"/../Xspec/src/manager/model.dat"        
    
        with open(input_file, 'r') as file:
            file_content = file.read()
        
        # Split the content by empty lines to separate different models
        model_sections = file_content.strip().split('\n\n')        
        models_info = {}
        
        for section in model_sections:           
            lines = section.strip().split('\n')        
            if not lines:
                continue
        
            first_line_parts = lines[0].split()
            model_name = first_line_parts[0].lower()
            model_type = first_line_parts[5] 
        
            parameters = {}   
            #Parse each parameter line, ignoring the first one (since it contains the model definition)
            for line in lines[1:]:
                parts = line.split()
                #All the exception catches are to deal with non-standard formatting like empty lines, 
                #switch parameters without bounds, weird formatting in the units, etc
                try: 
                    param_name = parts[0]
                    unit = parts[1]
                except IndexError:
                    continue            
                if param_name[0] == "$" or param_name[0] == "*":
                    param_name = param_name.strip('$*')
                    unit = "n/a"
                    try:
                        value = float(parts[1])
                        min_val = None
                        max_val = None
                    except ValueError:
                        try:
                            value = float(parts[2])
                            min_val = None
                            max_val = None
                        except ValueError:
                            value = float(parts[3])
                            min_val = None
                            max_val = None
                elif unit == '"':    
                    unit = "n/a"
                    value = float(parts[3])
                    min_val = float(parts[4])
                    max_val = float(parts[7])
                else:
                    try:
                        value = float(parts[2])
                        min_val = float(parts[3])
                        max_val = float(parts[6])
                    except ValueError:
                        unit = parts[1]+parts[2]
                        value = float(parts[3])
                        min_val = float(parts[4])
                        max_val = float(parts[7])                
                unit = unit.strip('"')
                parameters[param_name] = {
                    'value': value,
                    'min': min_val,
                    'max': max_val,
                    'unit': unit
                }       
            if model_type == 'add':
                parameters['norm'] = {
                    'value': 1,
                    'min': 0,
                    'max': 1e20,
                    'unit': "n/a"
                }                     
            models_info[model_name] = {
                'type': model_type,
                'parameters': parameters
            }
        return models_info    

    def print_model_info(self):
        print()
        print("Loaded Xspec models:")
        for component, details in self.models_info.items():
            print(f"{component}:")
            print(f"  type: {details['type']}")
            print("  parameters:")
            for param, values in details['parameters'].items():
                value_str = ', '.join(f"{key}: {value}" for key, value in values.items())
                print(f"    {param}: {value_str}")
            print()
        
    def add_model(self, func):
        func_name = func.__name__.rstrip('_')

        #sort out model parameters
        self.models_info[func_name] = self._all_info[func_name] 
    
        lib_func = getattr(self.lib, f"{func_name}_")
        lib_func.argtypes = [
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_float)
        ]
        lib_func.restype = None

        if self.models_info[func_name]['type'] == "add":        
            @wraps(func)
            def wrapper(ear, params):
                ne = len(ear) - 1
                photar = np.zeros(ne, dtype=np.float32)
                photer = np.zeros(ne, dtype=np.float32)
        
                lib_func(
                    ear.ctypes.data_as(ct.POINTER(ct.c_float)),
                    ct.byref(ct.c_int(ne)),
                    params.ctypes.data_as(ct.POINTER(ct.c_float)),
                    ct.byref(ct.c_int(1)),
                    photar.ctypes.data_as(ct.POINTER(ct.c_float)),
                    photer.ctypes.data_as(ct.POINTER(ct.c_float))
                )
                return photar*params[-1]
        elif self.models_info[func_name]['type'] == "mul":        
            @wraps(func)
            def wrapper(ear, params):
                ne = len(ear) - 1
                photar = np.zeros(ne, dtype=np.float32)
                photer = np.zeros(ne, dtype=np.float32)
        
                lib_func(
                    ear.ctypes.data_as(ct.POINTER(ct.c_float)),
                    ct.byref(ct.c_int(ne)),
                    params.ctypes.data_as(ct.POINTER(ct.c_float)),
                    ct.byref(ct.c_int(1)),
                    photar.ctypes.data_as(ct.POINTER(ct.c_float)),
                    photer.ctypes.data_as(ct.POINTER(ct.c_float))
                )
                return photar
        elif self.models_info[func_name]['type'] == "con":  
            @wraps(func)
            def wrapper(ear, params, seed):
                ne = len(ear) - 1
                seed = np.array(seed,dtype = np.float32)
                photer = np.zeros(ne, dtype = np.float32)

                lib_func(
                    ear.ctypes.data_as(ct.POINTER(ct.c_float)),
                     ct.byref(ct.c_int(ne)),
                     params.ctypes.data_as(ct.POINTER(ct.c_float)),
                     ct.byref(ct.c_int(1)),
                     seed.ctypes.data_as(ct.POINTER(ct.c_float)),
                     photer.ctypes.data_as(ct.POINTER(ct.c_float))
                )
                return seed
                
        # Attach the wrapper to the class 
        setattr(self, func_name, wrapper)        
        return func

    #def check_params_values(self,model_name,params):
    
    def load_models(self, models):
        for model_name, model_func in models.items():
            # Apply the decorator to each model function
            self.add_linear_model(model_func)
