import sys
import ctypes as ct
import os.path
import numpy as np
import matplotlib.pyplot as plt

type_float_p = ct.POINTER(ct.c_float)
type_int_p = ct.POINTER(ct.c_int)

#generic wrapper for multiplicative and additive models 
def xspec_wrap(ear, params, func):
    '''
    Takes:

    ear   : numpy array of energies
    params: array of parameters (double)

    Returns:

    photar: numpy.array (double)
    '''

    ne = len(ear) - 1

    photar = np.zeros(ne, dtype = np.float32)
    photer = np.zeros(ne, dtype = np.float32)

    func(ear.ctypes.data_as(type_float_p),
         ct.byref(ct.c_int(ne)),
         params.ctypes.data_as(type_float_p),
         ct.byref(ct.c_int(1)),
         photar.ctypes.data_as(type_float_p),
         photer.ctypes.data_as(type_float_p))

    return photar
    
#generic wrapper for convolution models
def xspec_convolve(ear, seed, params, func):
    '''
    Takes:

    ear   : numpy array of energies
    params: array of parameters (double)

    Returns:

    photar: numpy.array (double)
    '''

    ne = len(ear) - 1

    seed = np.array(seed,dtype = np.float32)
    photer = np.zeros(ne, dtype = np.float32)

    func(ear.ctypes.data_as(type_float_p),
         ct.byref(ct.c_int(ne)),
         params.ctypes.data_as(type_float_p),
         ct.byref(ct.c_int(1)),
         seed.ctypes.data_as(type_float_p),
         photer.ctypes.data_as(type_float_p))

    return seed

#missing models TBD: slimbh

def load_xspec_library(path):
    #load the xspec library file
    lib = ct.cdll.LoadLibrary(path+"/Xspec/x86_64-pc-linux-gnu-libc2.35/lib/libXSFunctions.so")
    #call fninit ti initialize abundances etc
    init_call = lib.fninit_
    init_call()
    
    #define all the models - additive models first 
    agnsed_call = lib.agnsed_
    agnsed_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                            type_int_p, type_float_p, type_float_p]
    agnsed_call.restype  = None
    def xspec_agnsed(ear, params):
        return xspec_wrap(ear, params, agnsed_call)  
        
    agnslim_call = lib.agnslim_
    agnslim_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                             type_int_p, type_float_p, type_float_p]
    agnslim_call.restype  = None
    def xspec_agnslim(ear, params):
        return xspec_wrap(ear, params, agnslim_call)      
        
    apec_call = lib.apec_
    apec_call.argtypes = [type_float_p,  type_int_p, type_float_p, 
                          type_int_p, type_float_p, type_float_p]
    apec_call.restype  = None
    def xspec_apec(ear, params):
        return xspec_wrap(ear, params, apec_call) 
    
    bbodyrad_call = lib.bbodyrad_
    bbodyrad_call.argtypes = [type_float_p,  type_int_p, type_float_p, 
                              type_int_p, type_float_p, type_float_p]
    bbodyrad_call.restype  = None
    def xspec_bbodrad(ear, params):
        return xspec_wrap(ear, params, bbodyrad_call) 
    
    compps_call = lib.compps_
    compps_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                            type_int_p, type_float_p, type_float_p]
    compps_call.restype  = None
    def xspec_compps(ear, params):
        return xspec_wrap(ear, params, compps_call)  
        
    diskbb_call = lib.diskbb_
    diskbb_call.argtypes = [type_float_p,  type_int_p, type_float_p, 
                            type_int_p, type_float_p, type_float_p]
    diskbb_call.restype  = None
    def xspec_diskbb(ear, params):
        return xspec_wrap(ear, params, diskbb_call) 
    
    diskir_call = lib.diskir_
    diskir_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                           type_int_p, type_float_p, type_float_p]
    diskir_call.restype  = None    
    def xspec_diskir(ear, params):
        return xspec_wrap(ear, params, diskir_call)  
    
    eqpair_call = lib.eqpair_
    eqpair_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                            type_int_p, type_float_p, type_float_p]
    eqpair_call.restype  = None
    def xspec_eqpair(ear, params):
        return xspec_wrap(ear, params, eqpair_call)  
    
    kerrbb_call = lib.kerbb_
    kerrbb_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                             type_int_p, type_float_p, type_float_p]
    kerrbb_call.restype  = None    
    def xspec_nthcomp(ear, params):
        return xspec_wrap(ear, params, kerrbb_call) 
    
    nthcomp_call = lib.nthcomp_
    nthcomp_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                             type_int_p, type_float_p, type_float_p]
    nthcomp_call.restype  = None    
    def xspec_nthcomp(ear, params):
        return xspec_wrap(ear, params, nthcomp_call)  
        
    slimbh_call = lib.slimbh_
    slimbh_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                             type_int_p, type_float_p, type_float_p]
    slimbh_call.restype  = None    
    def xspec_nthcomp(ear, params):
        return xspec_wrap(ear, params, slimbh_call) 
        
    #then multiplicative models
    redden_call = lib.redden_
    redden_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                            type_int_p, type_float_p, type_float_p]
    redden_call.restype  = None        
    def wrap_redden(ear, params):
        return xspec_wrap(ear, params, redden_call)  
    
    tbabs_call = lib.tbabs_
    tbabs_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                           type_int_p, type_float_p, type_float_p]
    tbabs_call.restype  = None        
    def wrap_tbabs(ear, params):
        return xspec_wrap(ear, params, tbabs_call)          
        
    #then convolution models 
    reflect_call = lib.reflect_
    reflect_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                             type_int_p, type_float_p, type_float_p]
    reflect_call.restype  = None    
    def xspec_reflect(ear, seed, params):
        return xspec_convolve(ear, seed, params, reflect_call)  
    
    simpl_call = lib.simpl_
    simpl_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                           type_int_p, type_float_p, type_float_p]
    simpl_call.restype  = None    
    def xspec_simpl(ear, seed, params):
        return xspec_convolve(ear, seed, params, simpl_call)  
    
    thcomp_call = lib.thcompf_
    thcomp_call.argtypes = [type_float_p, type_int_p, type_float_p, 
                            type_int_p, type_float_p, type_float_p]
    thcomp_call.restype  = None    
    def xspec_thcomp(ear, seed, params):
        return xspec_convolve(ear, seed, params, thcomp_call)            
