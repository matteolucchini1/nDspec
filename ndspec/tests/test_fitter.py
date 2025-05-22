import sys
import os
import warnings
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__/ndspec/'))))

from lmfit import Model as LM_Model
from lmfit import Parameters as LM_Parameters

from ndspec.Response import ResponseMatrix
from ndspec.FitPowerSpectrum import FitPowerSpectrum
from ndspec.FitTimeAvgSpectrum import FitTimeAvgSpectrum
from ndspec.FitCrossSpectrum import FitCrossSpectrum

import pytest


def ones_model(len):
    return np.ones(len)

def cross_const(energs,freqs):
    n_energs = len(energs)
    n_freqs = len(freqs)
    model = np.ones((n_freqs,n_energs))
    return model

class TestResponse(object):

    @classmethod
    def setup_class(cls):
        #set up the response to be used 
        rmffile = os.getcwd()+"/ndspec/tests/data/xrt.rmf"
        arffile = os.getcwd()+"/ndspec/tests/data/xrt.arf"
        cls.response = ResponseMatrix(rmffile)
        cls.response.load_arf(arffile)
        
        new_channels = np.linspace(cls.response.emin[0],cls.response.emax[-1],6)
        cls.rebin_matrix = cls.response.rebin_channels(new_channels[:-1],new_channels[1:])
        new_grid = 0.5*(cls.rebin_matrix.emax+cls.rebin_matrix.emin)
        new_width = cls.rebin_matrix.emax-cls.rebin_matrix.emin
        cls.new_edges = np.append(new_grid-0.5*new_width,new_grid[-1]+0.5*new_width[-1])
        
        #set up data+fitter to test the PSD
        dummy_psd = np.ones(10)
        freqs = np.linspace(1,10,10)
        cls.test_psd = FitPowerSpectrum()
        cls.test_psd.set_data(dummy_psd,0.1*dummy_psd,freqs)
        
        #set up data+fitter to test the time averaged spectrum 
        cls.test_spec = FitTimeAvgSpectrum()
        cls.test_spec.set_data(cls.response,os.getcwd()+"/ndspec/tests/data/xrt.fak")
        
        #set up data+fitter to test the cross spectrum 
        cross_freqs = np.linspace(0.2,1.0,5)
        cls.test_cross = FitCrossSpectrum()
        cls.test_cross.set_coordinates("polar")
        cls.test_cross.set_product_dependence("energy")

        dummy_mods = np.ones((4,5))
        dummy_phase = np.ones((4,5))

        dummy_cross = np.append(dummy_mods.flatten(),dummy_phase.flatten())
        dummy_cross_err = 0*dummy_cross

        cls.test_cross.set_data(cls.rebin_matrix,
                                [new_channels[0],new_channels[-1]],cls.new_edges,
                                dummy_cross,dummy_cross_err,
                                freq_bins=cross_freqs,
                                time_res=0.1,seg_size=10)
        
        #set the objects to test noticing/ignoring ranges         
        cls.test_select = FitCrossSpectrum()
        cls.test_select.set_coordinates("lags")
        cls.test_select.set_product_dependence("energy")
        
        cls.dummy_data = np.array([[1,2,3,4,5],
                                   [6,7,8,9,10],
                                   [11,12,13,14,15],
                                   [16,17,18,19,20]]).flatten()
        dummy_err = 0*cls.dummy_data.flatten()

        cls.test_select.set_data(cls.rebin_matrix,
                             [new_channels[0],new_channels[-1]],cls.new_edges,
                             cls.dummy_data,dummy_err,
                             freq_bins=cross_freqs,
                             time_res=0.1,seg_size=10)
        
        return 
       
    def test_psd_eval(self):
        psd_model = LM_Model(ones_model)
        model_parameters = LM_Parameters()
        model_parameters.add_many(('len', 10, False, 1, 20))

        self.test_psd.set_model(psd_model)
        self.test_psd.set_params(model_parameters)
        test_residuals = self.test_psd.get_residuals("delchi")
        assert(np.allclose(test_residuals[0],np.zeros(10)))
     
    def test_spec_eval(self):
        spec_model = LM_Model(ones_model)
        model_parameters = LM_Parameters()
        model_parameters.add_many(('len', 2400, False, 1, 3000))

        self.test_spec.set_model(spec_model,params=model_parameters)
        test_residuals = self.test_spec.get_residuals("ratio")
        #ignore the bins that are nan/inf because of the swift response 
        n_dof = self.test_spec.n_chans - 1 - 27
        test_stat = np.sum(test_residuals[0][27:-1])/n_dof
        #tolerance for Poisson noise in the simulated spectrum
        tol = 2e-3
        assert(np.allclose(test_stat,1,rtol=tol))
        
    def test_cross_eval(self):
        cross_model = LM_Model(cross_const,independent_vars=['energs','freqs'])
        cross_pars = LM_Parameters()

        self.test_cross.set_model(cross_model,model_type="cross")
        self.test_cross.set_params(cross_pars)
        test_model = self.test_cross.eval_model(fold=False)
        assert(np.allclose(test_model,self.test_cross.data))
        
    def test_select_bins(self):
        #test ignore frequencies:
        self.test_select.ignore_frequencies(0,0.4)
        self.test_select.ignore_frequencies(0.8,1.0)
        known_data = np.array([6,7,8,9,10,11,12,13,14,15])
        assert(np.allclose(known_data,self.test_select.data))

        #test notice frequencies
        self.test_select.notice_frequencies(0,0.4)
        self.test_select.notice_frequencies(0.8,1.2)
        assert(np.allclose(self.dummy_data,self.test_select.data))

        #test ignore energies
        self.test_select.ignore_energies(0,self.new_edges[1])
        self.test_select.ignore_energies(self.new_edges[-2],self.new_edges[-1])
        known_data = np.array([2,3,4,7,8,9,12,13,14,17,18,19])
        assert(np.allclose(known_data,self.test_select.data))

        #test notice energies
        self.test_select.notice_energies(0,self.new_edges[1])
        self.test_select.notice_energies(self.new_edges[-2],self.new_edges[-1])
        assert(np.allclose(self.dummy_data,self.test_select.data))

        #test both together
        self.test_select.ignore_frequencies(0,0.4)
        self.test_select.ignore_frequencies(0.8,1.0)
        self.test_select.ignore_energies(0,self.new_edges[1])
        self.test_select.ignore_energies(self.new_edges[-2],self.new_edges[-1])
        known_data = np.array([7,8,9,12,13,14])
        assert(np.allclose(known_data,self.test_select.data))
