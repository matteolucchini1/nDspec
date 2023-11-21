import numpy as np
import copy
from astropy.io import fits
#note: enable an option to switch between jax and numpy as the user wants
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
fi = 22
plt.rcParams.update({'font.size': fi-5})

colorscale = pl.cm.PuRd(np.linspace(0.,1.,5))

#note: the 2d plots are a little messed up because the bin edges aren't quite right,
#but the 3d should be fine

class ResponseMatrix(object):
    """This class has methods for:
    1) Load either a response matrix either from an OGIP-compliant .rsp FITS,
    or from an rmf and an arf file
    2) Convert the OGIP format into a matrix for the response
    3) Re-bin a given response to speed up convolving it with a model
    4) Multiply the response matrix with a model (in the form of another matrix)
    5) Plot the response, rmf or arf
    6) Create a diagonal matrix (e.g. for use with non X-ray instruments)
    
    Parameters
    ----------
    rmf_filepath: str
        The path to the response file to load 
    
    Attributes
    ----------
    channels: numpy.array(int)
        An array of integers representing each channel in the response. After rebinning 
        on a new grid, the number of channels is the number of bins in the new grid.
        
    numchan: int
        The legth of the channels array
    
    emin, emax: numpy.array(float)
        The arrays with the minimum/maximum energies in each channel in the channels 
        array. After rebinning on a new grid, it contains the minimum and maximum 
        energies of each bin in the new grid.
        
    energ_lo, energ_hi: numpy.array(float)
        The arrays with the minimum/maximum energies bins that sample the instrument
        rmf/arf
        
    numenerg: int
        The length of the energ_lo/energ_hi arrays
        
    resp_matrix: numpy.array(float x float)
        The instrument response in matrix form, with size (numenerg times numchan). 
        Only contains the redistribution matrix if no arf is loaded yet

    specresp: numpy.array(float)
        The instrument effective area as a function of energy as contained in the arf
        file
        
    has_arf: bool
        A flag to check whether only a rmf file has been loaded, or a full rmf+arf 
        response. This typically depends from mission to mission.         
    
    Methods
    ----------
    
    __init__(str):
        Initializes the class from a path to an appropriate FITS file by calling
        _load_rmf()
        
    _load_rmf(str):
        Loads either an rmf file, or a full response file, and sets the class
        attributes as appropriate
        
    _read_matrix(numpy.array,numpy.array,numpy.array,numpy.arry):
        Converts the information in the n_grp, f_chan, n_chan and matrix columns
        of a response file into a (numenerg times numchan) matrix
        
    load_arf(str):
        Loads effective area from an OGIP-compliants arf  FITSfile and applies it 
        to the resp_matrix attribute
        
    rebin_response(numpy.array,numpy.array):
        Rebins the resp_matrix attribute over the "channels" axis, starting from two
        user-provided arrays that contain the lower and upper energy bounds of the 
        new grid. The new grid needs to be more coarse than the initial one.
        
    bounds_to_channels(numpy.array,numpy.array):
        Maps two energy arrays containing lower and upper bins of a new energy grid
        to the channels loaded in the class, and returns the indexes of each bin
        in the current grid. Required by the rebin_response method.
        
    plot_response:
        If an arf or full response file were loaded, plots the full instrument response
        in channel vs energy space. Otherwise, plots the rmf in channel vs energy space
    
    plot_arf:
        If an arf was loaded, plots the instrument effective area as a function of energy
    
    diagonal_matrix(int):
        Sets the resp_matrix attribute to be a diagonal matrix of given input size, with 
        diagonal values 1
    
    convolve_response(numpy.array):
        Multiplies the resp_matrix attribute by a model matrix, with units of energy in
        the y axis and any other quantity in the x axis. Returns an "instrument space"
        matrix with units of channels (re-binned or not) in the y axis, and the initial
        quantity in the x axis. The model units in the z axis can either be in specific 
        photon flux (dN/dE, or photons per unit time, per unit energy), or in the default
        Xspec format of specific photon flux normalized to each bin width - dN/dE*bin_width.
        Users can specify which with the ``norm'' parameter - the default norm="rate" assumes
        dN/dE units, norm="xspec" assumes dN/dE * dE units.       
    """ 
    
    def __init__(self, filepath):
        self._load_rmf(filepath)
        #add check to also load the arf if provided
        pass
    
    def _load_rmf(self,filepath):
        self.rmfpath = filepath 
        response = fits.open(filepath)
        # get all the extension names
        extnames = np.array([h.name for h in response])
        self.bounds = response["EBOUNDS"]
        #figure out the right table to use, and decide 
        #whether we are also loading the arf or not 
        #redo this check some matrices are weird
        if "MATRIX" in extnames:
            h = response["MATRIX"]
            self.has_arf = False
            print("Arf missing, please load it")
        elif "SPECRESP MATRIX" in extnames:
            h = response["SPECRESP MATRIX"]
            self.has_arf = True
            print("Response file includes arf")
        #grab the relevant info from the fits file, store it in the channels_info/data/hdr objects,
        #and then close the fits file    
        channel_info = self.bounds.data
        data = h.data
        hdr = h.header
        if hdr["HDUCLASS"] != "OGIP":
            raise TypeError("File is not OGIP compliant")   
        response.close()
        
        self.emin = np.array(channel_info.field("E_MIN"))
        self.emax = np.array(channel_info.field("E_MAX"))
        self.channels = np.array(channel_info.field("CHANNEL"))
        self.numchan = len(self.channels)        

        self.energ_lo = np.array(data.field("ENERG_LO"))
        self.energ_hi = np.array(data.field("ENERG_HI"))
        self.numenerg = len(self.energ_lo)
        
        #store the rest of the information we need to convert the response 
        #to matrix format into arrays
        n_grp = np.array(data.field("N_GRP"))
        f_chan = np.array(data.field("F_CHAN"))
        n_chan = np.array(data.field("N_CHAN"))
        matrix = np.array(data.field("MATRIX"))
        
        self.resp_matrix = self._read_matrix(n_grp,f_chan,n_chan,matrix)
        return
        
    def _read_matrix(self,n_grp,f_chan,n_chan,matrix):
        #start with an empty matrix - we need to figure out 
        #which elements from the FITS file are not zero. These are the only values
        #reported in the FITS file, where the matrix has format 
        #(number of successive energy bins with no empty values n_grp) x
        #(number of bins that are not empty n_chan) x energy
        #we are trying to convert this to a matrix with format channel x energy
        resp_matrix = np.zeros((self.numenerg,self.numchan))
        #loop over the detector energies over which the rmf is binned 
        for j in range(self.numenerg):
            i = 0
            #loop over the number of channels in each channel set of consecutive bins with
            #no empty values
            for k in range(n_grp[j]):
                #Sometimes there are more than one groups of entries per row
                #As a result, we loop over the groups and assign the matrix values 
                #in the appropariate channel range as below:
                if any(m>1 for m in n_grp):     
                    for l in range(f_chan[j][k]+1,n_chan[j][k]+f_chan[j][k]):
                        i = i + 1
                        resp_matrix[j][l] = resp_matrix[j][l] + matrix[j][i]
                #In this case, the length of j-th row of the "Matrix" array is n_chan[j]+f_chan[j]
                #corresponding to channel indexes f_chan[j]+1 to n_chan[j]+f_chan[j]
                #so we set those values in coordinates j,l in the matrix array resp_matrix:
                else:
                    for l in range(f_chan[j]+1,n_chan[j]+f_chan[j]):
                        i = i + 1
                        resp_matrix[j][l] = resp_matrix[j][l] + matrix[j][i]  
        return resp_matrix
    
    #tbd: add option to load arf from the normal initialization 
    def load_arf(self,filepath):       
        self.arfpath = filepath
        self.has_arf = True
        response = fits.open(filepath)
        extnames = np.array([h.name for h in response])
        h = response["SPECRESP"]
        data = h.data
        hdr = h.header
        if hdr["HDUCLASS"] != "OGIP":
            raise TypeError("File is not OGIP compliant")   
        response.close()
        arf_emin = np.array(data.field("ENERG_LO"))
        arf_emax = np.array(data.field("ENERG_HI"))        
        if np.allclose(arf_emin,self.energ_lo) == False or np.allclose(arf_emax,self.energ_hi) == False:
            raise ValueError("Energy grids in rmf and arf do not match")
        
        self.specresp = np.array(data.field("SPECRESP"))
        if "EXPOSURE" in list(hdr.keys()):
            self.exposure = hdr["EXPOSURE"]
        else:
            self.exposure = 1.0
        
        full_resp = np.zeros((self.numenerg,self.numchan))
        for k in range(self.numchan):
            for j in range(self.numenerg):
                self.resp_matrix[j][k] = self.resp_matrix[j][k]*self.specresp[j]*self.exposure
        print("Arf loaded")
        return 
    

    #tbd: in the tutorial add an example of tryign to rebin in energy rather than channel
    #and show that it is dangerous
    def rebin_response(self,new_bounds_lo,new_bounds_hi):
        if new_bounds_lo[0] < self.emin[0]:
            raise ValueError("New channel grid below lower limit of existing one")
        if  new_bounds_hi[len(new_bounds_hi)-1] > self.emax[self.numchan-1]:
            raise ValueError("New channel grid above upper limit of existing one")
        if len(new_bounds_lo) != len(new_bounds_hi):
            raise TypeError("Lower and upper bounds of new channel grid have different size")
        if len(new_bounds_lo) > self.numchan:
            raise TypeError("You can not rebin to a finer channel grid")
        
        new_channels_lo,new_channels_hi = self.bounds_to_channels(new_bounds_lo,new_bounds_hi)
        rebinned_response = np.zeros((self.numenerg,len(new_channels_lo)))
        for j in range(self.numenerg):
            rebinned_response[j,:] = rebin_array((self.channels[0:self.numchan-1],
                                                  self.channels[1:self.numchan]),
                                                 (new_channels_lo,
                                                  new_channels_hi),
                                                 self.resp_matrix[j,:])
        
        bin_resp = copy.copy(self)
        bin_resp.emin = new_bounds_lo
        bin_resp.emax = new_bounds_hi
        bin_resp.numchan = len(new_bounds_lo)
        bin_resp.channels = np.linspace(0,bin_resp.numchan-1,bin_resp.numchan)
        bin_resp.resp_matrix = rebinned_response

        return bin_resp

    def bounds_to_channels(self,new_lo,new_hi):
        return_lo = np.zeros(len(new_lo))
        return_hi = np.zeros(len(new_hi))
        for i in range(len(new_lo)):
            #find the channel numbers corresponding to the start/end of each bin
            index_lo = np.digitize(new_lo[i],self.emin)
            index_hi = np.digitize(new_hi[i],self.emax)
            return_lo[i] = index_lo
            return_hi[i] = index_hi
        return return_lo,return_hi

    def convolve_response(self,model_input,norm="rate"):
        if np.shape(self.resp_matrix)[0] != np.shape(model_input)[0]:
            raise TypeError("Model energy grid has a different size from response")    
        if norm == "rate":
            bin_widths = self.energ_hi-self.energ_lo
            renorm_model = np.multiply(np.transpose(model_input),bin_widths)
            conv_model = np.matmul(renorm_model,self.resp_matrix)
        elif norm == "xspec":
            conv_model = np.matmul(model_input,self.resp_matrix)
        else:
            raise ValueError("Please specify units of either count rate or count rate normalized to bin width")
        return np.transpose(conv_model)

    def plot_response(self,plot_type="channel"):
        fig = plt.figure(figsize=(9.,7.5))
        
        if plot_type == "channel":
            x_axis = self.channels
            plt.xlabel("Channel")
        elif plot_type == "energy":
            x_axis = (self.emax+self.emin)/2.
            plt.xlabel("Bounds (keV)")
        else:
            raise TypeError("Specify either channel or energy for the x-axis")
                
        energy_array = (self.energ_hi+self.energ_lo)/2.
        plt.pcolor(x_axis,energy_array,np.log10(self.resp_matrix),
                   cmap="PuRd",shading='auto',linewidth=0,rasterized=True)
        plt.ylabel("Energy (keV)")
        plt.title("log10(Response)")
        plt.show()
        return
        
    def plot_arf(self,plot_scale="log"):
        #tbd: only allow this to happen if specresp is defined
        energy_array = (self.energ_hi+self.energ_lo)/2.
        fig = plt.figure(figsize=(9.,7.5))
        plt.plot(energy_array,self.specresp,linewidth=2.5,color=colorscale[3])
        plt.xlabel("Energy (keV)")
        plt.ylabel("Effective area")
        plt.yscale("log",base=10)
        if plot_scale == "log":
            plt.xscale("log",base=10)
        elif plot_scale != "lin":
            raise TypeError("Please specify either linear (lin) or logarithmic (log) x scale") 
        plt.show()
        return
        
    def diagonal_matrx(self,num):
        #tbd: figure out a less silly way to handle this
        #maybe call automatically if stuff fails
        diag_resp = np.diag(np.ones(num))
        return diag_resp            
              
    def unfold_response(self):
        print("TBD once the data side is complete")
        
def rebin_array(array_start,array_end,array):
    return_array = np.zeros(len(array_end[0]))
    for i in range(len(array_end[0])):
        #find the indexes of the bins in the old arrays that will go to the new one
        index_lo = np.digitize(array_end[0][i],array_start[0],right=True)
        index_hi = np.digitize(array_end[1][i],array_start[1])
        #first: calculate the contribution excluding the bin edges
        #loop over indexes of the incoming bins and do a bin-width weighted average
        #tbd: write clearer warning in case of index_lo = index_hi
        if index_lo == index_hi:
            raise IndexError("Outgoing bin "+str(i)+" has just one incoming bin, check energy grids")
        for k in range(index_lo,index_hi):            
            lower = np.max((array_start[0][k],array_end[0][i]))
            upper = np.min((array_start[1][k],array_end[1][i]))
            return_array[i] = return_array[i] + array[k]*(upper-lower)
        #second: include the contribution from the bin edges in order to renormalize correctly
        lower = array_end[0][i]
        upper = array_start[0][index_lo]
        return_array[i] = return_array[i] + array[index_lo-1]*(upper-lower)
        lower = array_start[1][index_hi-1]
        upper = array_end[1][i]     
        return_array[i] = return_array[i] + array[index_hi+1]*(upper-lower)                       
        return_array[i] = return_array[i]/(array_end[1][i]-array_end[0][i])
    return return_array
