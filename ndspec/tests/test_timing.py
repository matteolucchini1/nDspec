import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__/ndspec/'))))

import ndspec.Timing as timing 
import ndspec.models as models

from pyfftw.interfaces.numpy_fft import (
    fft,
    fftfreq,
)
import pytest
import warnings
from astropy.io import fits

#tbd: separate in test PSD and test Cross 
class TestTiming(object):

    @classmethod
    def setup_class(cls):
        cls.time_res_fft = 2500+1
        cls.times_fft = np.linspace(0.001,1.e3,cls.time_res_fft)
        time_bin = np.diff(cls.times_fft)[0]
        freqs_all = fftfreq(cls.time_res_fft,time_bin)
        cls.freqs_fft = freqs_all[freqs_all>0]
 
        cls.times_sinc = np.logspace(-1,3,500)
        cls.freqs_sinc = np.logspace(-3,-0.3,200)
        
        rmffile = "docs/data/nicer-rmf6s-teamonly-array50.rmf"
        with fits.open(rmffile, memmap=False) as response:
            extnames = np.array([h.name for h in response])
            h = response["MATRIX"]
            data = h.data
            energ_lo = np.array(data.field("ENERG_LO"))
            energ_hi = np.array(data.field("ENERG_HI"))
            cls.energies = 0.5*(energ_lo+energ_hi)

        cls.sin_wave = np.sin(cls.times_sinc)

        cls.lorentz_fft = models.lorentz(cls.freqs_fft,
                                         np.array([0.0075,0.1,0.1])) + \
                          models.lorentz(cls.freqs_fft,
                                         np.array([0.05,0.1,0.1]))               
        cls.lorentz_sinc = models.lorentz(cls.freqs_sinc,
                                         np.array([0.0075,0.1,0.1])) + \
                          models.lorentz(cls.freqs_sinc,
                                         np.array([0.05,0.1,0.1]))                     

        cls.bbflash_fft, bbspec, bbpulse = models.bbody_bkn(cls.times_fft,
                                           cls.energies,
                                           np.array([1.,0.5,1.,-1.5,3.,-0.05]))
        cls.bbflash_sinc, bbspec, bbpulse = models.bbody_bkn(cls.times_sinc,
                                           cls.energies,
                                           np.array([1.,0.5,1.,-1.5,3.,-0.05]))                                           
        
        cls.pivoting_fft = models.pivoting_pl(cls.freqs_fft,
                                  cls.energies,
                                  np.array([1.,-1.9,0.2,-0.5,-0.5,1e-3]))
        cls.pivoting_sinc = models.pivoting_pl(cls.freqs_sinc,
                          cls.energies,
                          np.array([1.,-1.9,0.2,-0.5,-0.5,1e-3]))
                                        
        return 
        
    def test_powerspectrum_init(self):
    #tbd: separate these 
        with pytest.raises(ValueError):
            wrong_times = np.array([0,1,2,3,4,6,5,7,8,9,10])
            powerspectrum = timing.PowerSpectrum(times=wrong_times)
        with pytest.raises(TypeError):
            wrong_frequencies = np.array(([3,2],[1,3]))
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                              freqs = wrong_frequencies,
                                              method='sinc')
        with pytest.raises(ValueError):
            wrong_frequencies = np.array([0,1,2,3,4,6,5,7,8,9,10])
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                              freqs = wrong_frequencies,
                                              method='sinc')
        with pytest.warns(UserWarning):
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              method='wrong')            
#tbd: test both the attribute error in set_frequency(like here) and transform 
        with pytest.raises(AttributeError):
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              method='sinc')
            powerspectrum.method='wrong'
            powerspectrum.power_spec=self.lorentz_sinc
            powerspectrum.rebin_frequency(self.freqs_sinc)

    def test_sinc_exists(self):
        with pytest.raises(AttributeError):
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              method='sinc')
            del powerspectrum.irf_sinc_arr 
            powerspectrum.compute_psd(self.sin_wave)   
            
    def test_method_exists(self):                                    
        with pytest.raises(AttributeError):
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              method='sinc')
            del powerspectrum.method 
            powerspectrum.compute_psd(self.sin_wave)  
            
    def test_psd_array_size(self):
        with pytest.raises(TypeError):
            powerspectrum = timing.PowerSpectrum(self.times_fft)
            powerspectrum.compute_psd(self.sin_wave)
            
    def test_interpolate_bounds(self):
        powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                      freqs = self.freqs_sinc,
                                      method='sinc')
        powerspectrum.compute_psd(self.sin_wave)  
        new_grid_lo = 0.5*self.freqs_sinc
        new_grid_hi = 1.5*self.freqs_sinc
        with pytest.raises(ValueError):
            powerspectrum.rebin_frequency(new_grid_lo)
        with pytest.raises(ValueError):
            powerspectrum.rebin_frequency(new_grid_hi)

    def test_crossspectrum_init(self):
        with pytest.raises(ValueError):
            wrong_energs = np.array([0,1,2,3,4,6,5,7,8,9,10])
            crossspectrum = timing.CrossSpectrum(self.times_fft,
                                                 energ=wrong_energs)   
        with pytest.raises(ValueError):
            wrong_times = np.array([0,1,2,3,4,6,5,7,8,9,10])
            crossspectrum = timing.CrossSpectrum(times=wrong_times,
                                                 energ=self.energies)
        with pytest.raises(TypeError):
            wrong_frequencies = np.array(([3,2],[1,3]))
            crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                              freqs = wrong_frequencies,
                                              energ=self.energies,
                                              method='sinc')
        with pytest.raises(ValueError):
            wrong_frequencies = np.array([0,1,2,3,4,6,5,7,8,9,10])
            crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                              freqs = wrong_frequencies,
                                              energ=self.energies,
                                              method='sinc')
        with pytest.warns(UserWarning):
            crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              energ=self.energies,
                                              method='wrong')            
        with pytest.raises(AttributeError):
            crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              energ=self.energies,
                                              method='sinc')
            crossspectrum.set_impulse(self.bbflash_sinc)
            crossspectrum.set_reference_lc(self.bbflash_sinc[0,:])
            crossspectrum.set_psd_weights(self.lorentz_sinc)
            crossspectrum.cross_from_irf()
            crossspectrum.method='wrong'
            crossspectrum.rebin_frequency(self.freqs_sinc)
        
    def test_set_psd_weights(self):
        with pytest.raises(TypeError):
            crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              energ=self.energies,
                                              method='sinc')
            crossspectrum.set_psd_weights(self.lorentz_fft)
        
    def test_set_impulse(self):
        crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                      freqs = self.freqs_sinc,
                                      energ=self.energies,
                                      method='sinc')
        with pytest.raises(TypeError):
            crossspectrum.set_impulse(self.bbflash_sinc[1:,:]) 
        with pytest.raises(TypeError):    
            crossspectrum.set_impulse(self.bbflash_fft)
                                         
    def test_set_transfer(self):
        crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                      freqs = self.freqs_sinc,
                                      energ=self.energies,
                                      method='sinc')
        with pytest.raises(TypeError):
            crossspectrum.set_transfer(self.pivoting_sinc[1:,:]) 
        with pytest.raises(TypeError):    
            crossspectrum.set_transfer(self.pivoting_fft)
                                                     
    def test_set_reference(self):
        crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                      freqs = self.freqs_sinc,
                                      energ=self.energies,
                                      method='sinc')
        with pytest.raises(AttributeError):
            crossspectrum.set_reference_energ([1.0,2.])         
        crossspectrum.set_impulse(self.bbflash_sinc)
        with pytest.raises(ValueError): 
            crossspectrum.set_reference_energ([300.,400.])  
        with pytest.raises(ValueError): 
            crossspectrum.set_reference_energ([1.0,1.0001])  

    def test_cross_found(self):
        crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                  freqs = self.freqs_sinc,
                                  energ=self.energies,
                                  method='sinc')
        with pytest.raises(AttributeError):
            test = crossspectrum.real()
        with pytest.raises(AttributeError):
            test = crossspectrum.imag()
        with pytest.raises(AttributeError):
            test = crossspectrum.mod()    
        with pytest.raises(AttributeError):
            test = crossspectrum.phase()          
        with pytest.raises(AttributeError):
            test = crossspectrum.lag()
              
    def test_oned_range_errors(self):
        crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                  freqs = self.freqs_sinc,
                                  energ=self.energies,
                                  method='sinc')
        crossspectrum.set_impulse(self.bbflash_sinc)
        crossspectrum.set_reference_energ(self.bbflash_sinc[0,:])
        crossspectrum.set_psd_weights(self.lorentz_sinc)
        with pytest.raises(ValueError):
            error = crossspectrum.lag_frequency(int_bounds=[1.0,1.0001])  
        with pytest.raises(ValueError):
            error = crossspectrum.lag_frequency(int_bounds=[300.,400.])  
        crossspectrum.cross_from_irf()
        with pytest.raises(ValueError):
            error = crossspectrum.phase_energy(self.freqs_sinc[4],
                                               self.freqs_sinc[2])
                                               
 #   def test_cross_methods(self):
    #test that for an IRF the fft and sinc return similar things when interpolated
    #to the same grid 
