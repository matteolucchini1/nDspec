import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__/ndspec/'))))
from ndspec.Response import ResponseMatrix, rebin_array
import pytest
from xspec import *
import warnings

class TestRMF(object):

    @classmethod
    def setup_class(cls):
        cls.rmffile = "docs/data/nicer-rmf6s-teamonly-array50.rmf"
        cls.response = ResponseMatrix(cls.rmffile)
        cls.arffile = "docs/data/nicer-consim135p-teamonly-array50.arf"
        return

    def test_response_exists(self):
        response = ResponseMatrix(self.rmffile)
        self.response.load_arf(self.arffile)
   
    def test_rmf_has_correct_attributes(self):
        assert hasattr(self.response, "energ_lo"), "row missing in rmf file"
        assert hasattr(self.response, "energ_hi"), "row missing in rmf file"
        assert hasattr(self.response, "emin"), "row missing in rmf file"
        assert hasattr(self.response, "emax"), "row missing in rmf file"
        assert hasattr(self.response, "n_chans"), "row missing in rmf file"
        assert hasattr(self.response, "n_energs"), "row missing in rmf file"
        assert hasattr(self.response, "chans"), "row missing in rmf file"
        assert hasattr(self.response, "resp_matrix"), "row missing in rmf file"
        assert hasattr(self.response, "specresp"), "row missing in rmf file"
 
    def test_nicer_convolution_comparison(self):
        Xset.chatter = 0
        AllData.clear()
        AllModels.clear()
        placeholder = Spectrum(os.getcwd()+"/ndspec/tests/data/nicer_unbinned.pha")
        placeholder.response = self.rmffile
        placeholder.response.arf = self.arffile
        m1 = Model("powerlaw")
        m1.powerlaw.PhoIndex = 2.        
        Plot("ldata")
        modVals = Plot.model()        
        model_values = np.array(m1.values(1))
        model_energies = np.array(m1.energies(1))
        bin_widths = np.diff(model_energies)
        convolved_model = self.response.convolve_response(
                                        model_values,
                                        norm="xspec")
        convolved_rebinned = self.response.convolve_response(
                                           model_values/bin_widths,
                                           norm="rate")
        #the first 10 bins are ignored because there is a discrepancy with xpsec
        #but it has no impact on results since data below 0.1 keV is never used
        assert np.allclose(modVals[10:],convolved_model[10:],rtol=1e-3) == True
        assert np.allclose(modVals[10:],convolved_rebinned[10:],rtol=1e-3) == True

    def test_model_grid_match(self):
        wrong_gridsize_model = np.linspace(2,-2,self.response.n_energs-1)
        with pytest.raises(TypeError):
            wrong_grid_convolution = self.response.convolve_response(
                                                   wrong_gridsize_model)

    def test_rmf_ogip_error(self):
        wrong_rmf = os.getcwd()+"/ndspec/tests/data/nicer_notogip.rmf"
        with pytest.raises(TypeError):
            wrong_response = ResponseMatrix(wrong_rmf)
            
    def test_arf_ogip_error(self):
        wrong_arf = os.getcwd()+"/ndspec/tests/data/nicer_notogip.arf"
        right_response = ResponseMatrix(self.rmffile)
        with pytest.raises(TypeError):
            right_response.load_arf(wrong_arf)
   
    def test_matching_energy_grids(self):
        xrt_arf = os.getcwd()+"/ndspec/tests/data/xrt_wt.arf"
        nicer_response = ResponseMatrix(self.rmffile)
        with pytest.raises(ValueError):
            nicer_response.load_arf(xrt_arf)
            
    def test_rebin_grid(self):
        new_bounds_lo = np.array([0.1,0.3,0.6,1,1.5,2,2.13,2.21,2.3,2.4,2.52,
                                  2.6,2.9,3.,3.5,4.,4.12,4.25,4.4,4.54,5,
                                  5.5,5.75,6,6.1,6.3,6.4,6.8,7,7.2,7.3,7.35,7.6,
                                  8,9.5])
        new_bounds_hi = np.array([0.3,0.6,1,1.5,2,2.13,2.21,2.3,2.4,2.52,2.6,
                                  2.9,3.,3.5,4.,4.12,4.25,4.4,4.54,5,5.5,5.75,6,
                                  6.1,6.3,6.4,6.8,7,7.2,7.3,7.35,7.6,8,9.5,10.])
        with pytest.raises(ValueError):
            new_bounds_lo[0] = -0.01
            rebin = self.response.rebin_response(new_bounds_lo,new_bounds_hi)
        with pytest.raises(ValueError):
            new_bounds_lo[0] = 0.1
            new_bounds_hi[len(new_bounds_hi)-1] = 30.
            rebin = self.response.rebin_response(new_bounds_lo,new_bounds_hi)
        with pytest.raises(TypeError):
            new_bounds_hi[len(new_bounds_hi)-1] = 10.
            new_bounds_wrong = new_bounds_hi[1:]
            rebin = self.response.rebin_response(new_bounds_lo,new_bounds_wrong)
        with pytest.raises(TypeError):
            new_bounds_fine = np.linspace(0.1,10.,4001)
            new_bounds_fine_lo = new_bounds_fine[:len(new_bounds_fine)-1]
            new_bounds_fine_hi = new_bounds_fine[1:]
            rebin = self.response.rebin_response(new_bounds_fine_lo,new_bounds_fine_hi)
        #last test: energy bin sizes
        with pytest.raises(IndexError):
            new_bounds_lo[1] = 0.101
            new_bounds_hi[0] = new_bounds_lo[1]
            new_channels_lo,new_channels_hi = self.response._bounds_to_chans(
                                                            new_bounds_lo,
                                                            new_bounds_hi)
                                                            
            rebin_array((self.response.chans[0:self.response.n_chans-1],
                         self.response.chans[1:self.response.n_chans]),
                         (new_channels_lo, new_channels_hi),
                         self.response.resp_matrix[0,:])

    def test_response_plot(self):
        with pytest.raises(TypeError):
            self.response.plot_response(plot_type="wrong_type")
            
    def test_arf_plot(self):
        with pytest.raises(TypeError):
            self.response.plot_arf(plot_scale="wrong_scale")              
