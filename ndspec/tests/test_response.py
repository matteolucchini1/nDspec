import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__/ndspec/'))))
from ndspec.Response import ResponseMatrix
import pytest
from xspec import *

class TestResponse(object):

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

    #compare the convolution with the nicer response to that in xspec 
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
        assert np.allclose(modVals,convolved_model,rtol=1e-6) == True
        assert np.allclose(modVals,convolved_rebinned,rtol=1e-6) == True

    #compare the convolution with the nustar response to that in xspec        
    def test_nustar_convolution_comparison(self):
        nustar_fpma = ResponseMatrix(os.getcwd()+"/ndspec/tests/data/nustar_fpma.rmf")
        nustar_fpma.load_arf(os.getcwd()+"/ndspec/tests/data/nustar_fpma.arf") 
        Xset.chatter = 0
        AllData.clear()
        AllModels.clear()
        placeholder = Spectrum(os.getcwd()+"/ndspec/tests/data/nustar_fpma.pha")
        placeholder.response = os.getcwd()+"/ndspec/tests/data/nustar_fpma.rmf"
        placeholder.response.arf = os.getcwd()+"/ndspec/tests/data/nustar_fpma.arf"
        m1 = Model("powerlaw")
        m1.powerlaw.PhoIndex = 2.        
        Plot("ldata")
        modVals = Plot.model()        
        model_values = np.array(m1.values(1))
        model_energies = np.array(m1.energies(1))
        bin_widths = np.diff(model_energies)
        convolved_model = nustar_fpma.convolve_response(
                                      model_values,
                                      norm="xspec")
        convolved_rebinned = nustar_fpma.convolve_response(
                                         model_values/bin_widths,
                                         norm="rate")
        assert np.allclose(modVals,convolved_model,rtol=1e-6) == True
        assert np.allclose(modVals,convolved_rebinned,rtol=1e-6) == True

    #compare the convolution with the rxte response to that in xspec
    def test_rxte_convolution_comparison(self):
        rxte_pca = ResponseMatrix(os.getcwd()+"/ndspec/tests/data/rxte.rsp")
        Xset.chatter = 0
        AllData.clear()
        AllModels.clear()
        placeholder = Spectrum(os.getcwd()+"/ndspec/tests/data/rxte.pha")
        placeholder.response = os.getcwd()+"/ndspec/tests/data/rxte.rsp"
        m1 = Model("powerlaw")
        m1.powerlaw.PhoIndex = 2.        
        Plot("ldata")
        modVals = Plot.model()        
        model_values = np.array(m1.values(1))
        model_energies = np.array(m1.energies(1))
        bin_widths = np.diff(model_energies)
        convolved_model = rxte_pca.convolve_response(
                                   model_values,
                                   norm="xspec")
        convolved_rebinned = rxte_pca.convolve_response(
                                      model_values/bin_widths,
                                      norm="rate")
        assert np.allclose(modVals,convolved_model,rtol=1e-6) == True
        assert np.allclose(modVals,convolved_rebinned,rtol=1e-6) == True

    #compare the convolution with the xrt response to that in xspec
    def test_xrt_convolution_comparison(self):
        xrt_resp = ResponseMatrix(os.getcwd()+"/ndspec/tests/data/xrt.rmf")
        xrt_resp.load_arf(os.getcwd()+"/ndspec/tests/data/xrt.arf") 
        Xset.chatter = 0
        AllData.clear()
        AllModels.clear()
        placeholder = Spectrum(os.getcwd()+"/ndspec/tests/data/xrt.pha")
        placeholder.response = os.getcwd()+"/ndspec/tests/data/xrt.rmf"
        placeholder.response.arf = os.getcwd()+"/ndspec/tests/data/xrt.arf"
        m1 = Model("powerlaw")
        m1.powerlaw.PhoIndex = 2.        
        Plot("ldata")
        modVals = Plot.model()        
        model_values = np.array(m1.values(1))
        model_energies = np.array(m1.energies(1))
        bin_widths = np.diff(model_energies)
        convolved_model = xrt_resp.convolve_response(
                                   model_values,
                                   norm="xspec")
        convolved_rebinned = xrt_resp.convolve_response(
                                      model_values/bin_widths,
                                      norm="rate")
        assert np.allclose(modVals,convolved_model,rtol=1e-6) == True
        assert np.allclose(modVals,convolved_rebinned,rtol=1e-6) == True

    #compare the convolution with the xmm/epic response to that in xspec
    def test_xmm_convolution_comparison(self):
        xmm_resp = ResponseMatrix(os.getcwd()+"/ndspec/tests/data/xmm.rmf")
        xmm_resp.load_arf(os.getcwd()+"/ndspec/tests/data/xmm.arf") 
        Xset.chatter = 0
        AllData.clear()
        AllModels.clear()
        placeholder = Spectrum(os.getcwd()+"/ndspec/tests/data/xmm.pha")
        placeholder.response = os.getcwd()+"/ndspec/tests/data/xmm.rmf"
        placeholder.response.arf = os.getcwd()+"/ndspec/tests/data/xmm.arf"
        m1 = Model("powerlaw")
        m1.powerlaw.PhoIndex = 2.        
        Plot("ldata")
        modVals = Plot.model()        
        model_values = np.array(m1.values(1))
        model_energies = np.array(m1.energies(1))
        bin_widths = np.diff(model_energies)
        convolved_model = xmm_resp.convolve_response(
                                   model_values,
                                   norm="xspec")
        convolved_rebinned = xmm_resp.convolve_response(
                                      model_values/bin_widths,
                                      norm="rate")
        assert np.allclose(modVals,convolved_model,rtol=1e-6) == True
        assert np.allclose(modVals,convolved_rebinned,rtol=1e-6) == True

    #test that we can't convolve models with the wrong number of channels
    def test_model_grid_match(self):
        wrong_gridsize_model = np.linspace(2,-2,self.response.n_energs-1)
        with pytest.raises(TypeError):
            wrong_grid_convolution = self.response.convolve_response(
                                                   wrong_gridsize_model)

    #test that the code checks for the OGIP flag in the header of a rmf
    def test_rmf_ogip_error(self):
        wrong_rmf = os.getcwd()+"/ndspec/tests/data/nicer_notogip.rmf"
        with pytest.raises(TypeError):
            wrong_response = ResponseMatrix(wrong_rmf)

    #test that the code checks for the OGIP flag in the header of a arf            
    def test_arf_ogip_error(self):
        wrong_arf = os.getcwd()+"/ndspec/tests/data/nicer_notogip.arf"
        right_response = ResponseMatrix(self.rmffile)
        with pytest.raises(TypeError):
            right_response.load_arf(wrong_arf)
 
     #test that the code forces the ebounds to be the same in the rmf and arf
    def test_matching_energy_grids(self):
        xrt_arf = os.getcwd()+"/ndspec/tests/data/xrt_wt.arf"
        nicer_response = ResponseMatrix(self.rmffile)
        with pytest.raises(ValueError):
            nicer_response.load_arf(xrt_arf)
 
    #test that rebinning to weird grids raises the appropriate errors           
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
        #with pytest.raises(IndexError):
        #    new_bounds_lo[1] = 0.101
        #    new_bounds_hi[0] = new_bounds_lo[1]
       #     new_channels_lo,new_channels_hi = self.response._bounds_to_chans(
        #                                                    new_bounds_lo,
        #                                                    new_bounds_hi)
                                                            
        #    rebin_array((self.response.chans[0:self.response.n_chans-1],
         #                self.response.chans[1:self.response.n_chans]),
          #               (new_channels_lo, new_channels_hi),
           #              self.response.resp_matrix[0,:])

    def test_response_plot(self):
        with pytest.raises(TypeError):
            self.response.plot_response(plot_type="wrong_type")
            
    def test_arf_plot(self):
        with pytest.raises(TypeError):
            self.response.plot_arf(plot_scale="wrong_scale")              

    def test_exposure_time(self):
        current_exposure = self.response.exposure
        current_resp_grid = self.response.resp_matrix
        self.response.set_exposure_time(10*current_exposure)
        assert self.response.exposure == 10*current_exposure
        assert self.response.resp_matrix == 10*current_resp_grid
        
        with pytest.raises(TypeError):
            self.response.set_exposure_time("str")
    
    def test_ignore_energy_channels(self):
        #check that ignore energy channels works for channel indexing
        new_response = self.response.ignore_channels(low_chan=1)
        assert new_response.n_chans == self.response.n_chans-1
        new_response = self.response.ignore_channels(high_chan=1)
        assert new_response.n_chans == self.response.n_chans-1
        new_response = self.response.ignore_channels(low_chan=1,high_chan=3)
        assert new_response.n_chans == self.response.n_chans-2
        #check that ignore energy channels works for energy bounds
        new_response = self.response.ignore_channels(low_energy=0.1,
                                                     high_energy=2)
        assert np.any(new_response.emin == 0.1) != True
        assert np.any(new_response.emax == 2) != True
        
        with pytest.raises(ValueError):
            self.response.ignore_channels()
        with pytest.raises(ValueError):
            self.response.ignore_channels(low_chan=-1)
        with pytest.raises(ValueError):
            self.response.ignore_channels(high_chan=10000)
        with pytest.raises(TypeError):
            self.response.ignore_channels(low_chan="")
        with pytest.raises(TypeError):
            self.response.ignore_channels(high_chan="")
        with pytest.raises(TypeError):
            self.response.ignore_channels(low_energy="")
        with pytest.raises(TypeError):
            self.response.ignore_channels(high_energy="")
        
        
        return