import ctypes as ct
import numpy as np
from functools import wraps
import os

#note: this is not compatible with Relxill specifically, will need a different wrapper urgh
class XspecLibrary:
    def __init__(self, lib_path=None):
        # Default library path
        if lib_path is None:
            headas_path = os.environ.get("HEADAS")
            if headas_path:
                lib_path = headas_path + f"/../Xspec/{os.path.basename(headas_path)}/lib/libXSFunctions.so"
            else:
                raise EnvironmentError("HEADAS environment variable not set.")

        # Load the library
        self.lib = ct.cdll.LoadLibrary(lib_path)
           
    def initialize_heasoft(self):
        init_call = self.lib.fninit_
        init_call()

def add_linear_model(instance, func):
    # Strip trailing underscore from the function name
    func_name = func.__name__.rstrip('_')

    # Get the library function with the trailing underscore
    lib_func = getattr(instance.lib, f"{func_name}_")
    lib_func.argtypes = [
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float)
    ]
    lib_func.restype = None

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
        #note: xspec includes a normalization parameter by default
        #I will need two wrappers - one for multiplicative models which is this one
        #the other for additive models which renormalize based on the value of the last parameter
        #thank you xspec!
        return photar

    # Attach the wrapper to the instance with the name without the underscore
    setattr(instance, func_name, wrapper)
    return func

def register_models(instance, models):
    for model_name, model_func in models.items():
        # Apply the decorator to each model function
        add_linear_model(instance, model_func)
        
def parse_model_parameters(input_file=None):
    if input_file is None:
        headas_path = os.environ.get("HEADAS")
        input_file =  headas_path + f"/../Xspec/src/manager/model.dat"        

    with open(input_file, 'r') as file:
        file_content = file.read()
    
    # Split the content by empty lines to separate different models
    model_sections = file_content.strip().split('\n\n')
    
    # Dictionary to store the parsed information for each model
    models_data = {}
    
    for section in model_sections:
        # Split each section into lines
        lines = section.strip().split('\n')
    
        if not lines:
            continue
    
        # Extract the model name and type from the entries in the first line
        first_line_parts = lines[0].split()
        model_name = first_line_parts[0].lower()
        model_type = first_line_parts[5] 
    
        parameters = {}   
        # Process each parameter line, ignoring the first one (since it contains the model definition)
        for line in lines[1:]:
            parts = line.split()
            #ignore the empty lines that somehow only get parsed with non-standard (e.g. switch) parameter formats
            try: 
                param_name = parts[0]
                unit = parts[1]
            except IndexError:
                continue    
    
            #handle all the weird dedicated switch parameters that have one-off line structures 
            if param_name[0] == "$" or param_name[0] == "*":
                param_name = param_name.strip('$*')
                unit = " "
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
                #shift the indexes by one because the initial split call actually adds two lines 
                #if a model has " " to mark dimensionless units, one per each quotation mark
                unit = "n/a"
                value = float(parts[3])
                min_val = float(parts[4])
                max_val = float(parts[7])
            else:
                # Handle cases where the parameter is not in dimensionless units
                try:
                    value = float(parts[2])
                    min_val = float(parts[3])
                    max_val = float(parts[6])
                #handle cases with spaces in the parameter units, which shifts everything by one
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
    
        # Store the parameters and type in the models_data dictionary
        models_data[model_name] = {
            'type': model_type,
            'parameters': parameters
        }
    return models_data

#this returns only the model parameters for a given model
def return_model_parameters(model,input_file=None):
    if input_file is None:
        headas_path = os.environ.get("HEADAS")
        input_file =  headas_path + f"/../Xspec/src/manager/model.dat"     
    all_params = parse_model_parameters(input_file)
    selected_params = all_params[model]['parameters']
    return selected_params     
