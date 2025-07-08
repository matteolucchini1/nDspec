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
