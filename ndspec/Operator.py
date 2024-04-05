import numpy as np
from scipy.interpolate import interp1d

#import pyfftw
#from pyfftw.interfaces.numpy_fft import (
#    fft,
#    fftfreq,
#)

class nDspecOperator(object):
    """
    Generic class from which all nDspec operators are inherited. It contains 
    standardized, private methods to manipulate arrays by performing 
    e.g. interpolation, integration over a range or axis, etc. These can be
    called by each specific operator to perform physical calculations, such as 
    integrating over a Fourier frequency range to calculate a lag spectrum. This
    class is not to be instantiated by itself.
    """
    
    def __init__(self):
        pass

    def _interpolate(self,array,old_grid,new_grid,use_log=True,grid_tol=1e-4):
        """
        This method interpolats a one-dimensional input, defined over 
        some grid, over an arbitrary new range, provided that this new range is
        contained within "old_grid".
        
        Parameters:
        ----------
        array: np.array(float) 
            The input array to be interpolated
            
        old_grid: np.array(float)
            The grid of e.g. Fourier frequency, energy etc. over which "array"
            is defined.
            
        new_grid: np.array (float)
            The new grid over which "array" is to be interpolated. Its extremes
            need to be contained within old_grid; for safety this method does
            NOT perform extrapolation beyond the existing bounds. 
        
        Other parameters:
        ----------
        grid_told: float, default 1e-4
            Sets the precision to which  the boundaries of "new_grid" are set
            to be contained contained within "old_grid" to some small precision.
            This is a cautionary step to avoid some numerical issues which can 
            show up when using scipy interp1d.  

        use_log: bool, default True
            Switches between interpolating the input array, or the base 10 
            logarithm of the  input array, which is more accurate if the latter 
            varies by many orders of magnitude over the grid
        
        Returns
        ---------- 
        interp_array: np.array(float) 
            The values of the input "array", interpolated over the updated grid 
            "new_grid".
        """
              
        if (new_grid[0] < old_grid[0]):
            print(new_grid[0],old_grid[0])
            raise ValueError("New grid lower boundary exceeds old grid")
        if (new_grid[-1] > old_grid[-1]):
            print(new_grid[-1],old_grid[-1])
            raise ValueError("New grid upper boundary exceeds old grid")

        if (new_grid[0] == old_grid[0]):
            new_grid[0] = new_grid[0] + (new_grid[1]-new_grid[0])*grid_tol
        if (new_grid[-1] == old_grid[-1]):
            new_grid[-1] = new_grid[-1] - (new_grid[-1]-new_grid[-2])*grid_tol

        if use_log is True:
            interp_obj = interp1d(old_grid,np.log10(array))
            interp_array = np.power(10.,interp_obj(new_grid))
        else:
            interp_obj = interp1d(old_grid,array)
            interp_array = interp_obj(new_grid)
        
        return interp_array

    def _integrate_range(self,signal,array,arr_min,arr_max,axis):
        """ 
        This method integrates an input two-d array over a defined range and 
        aixs, using the numpy implementation of the trapezoid method. Note that 
        the bounds provided are included in the integration.
        A generalized method to deal with one- or multi-dimensional data will
        be implemented in the future.         
        
        Parameters:
        ----------
        signal: np.array(float,float)
            The (two-dimensional) input to be integrated over one of the two
            axis.
            
        array: np.array(float)
            The one-dimensional array over which to perform the integration.
        
        array_min, array_max: float
            The minimum and maximum values in the "array" input over which to
            perform the integration.
            
        axis: int
            The axis over which to integrate the input "signal". In the 
            convetion adopted throughout the software for typical products, 
            axis=1 means integrating over energy, axis=0 means integrating over 
            other dimensions (e.g. Fourier frequency, time, etc depending on the
            specific operator).
        
        Returns
        ----------     
        integral: float 
            The result of the integral.
        """    
        
        if (arr_min >= arr_max):
            raise ValueError("Lower integration bound higher than upper integration bound")       
        
        arr_range = np.where(np.logical_and(array>=arr_min,array<=arr_max))

        if (len(arr_range) == 0):
            raise TypeError("No bins found within the integration bounds")
        
        if (axis == 0):
            integral =  np.trapz(signal[arr_range,:],x=array[arr_range])
        elif (axis == 1):
            integral =  np.trapz(signal[:,arr_range],x=array[arr_range])
        else:
            raise ValueError("Incorrect axis specified")
        
        return integral

    def _grid_bounds_to_range(self,grid_lower_bounds,grid_upper_bounds):
        """
        One of five similar methods to compute quantities relevant to grid 
        arrays (such as bin widths or center points).
        
        This method takes two arrays containing the lower and upper bounds of 
        the grid, and returns a combined array which starts from the first lower
        bound in the grid, and ends at the last upper bound in the same grid. 
        Only grids with contigous bins with no gaps (as is typical of e.g. X-ray
        spectral data) are supported. 
        
        As an example, assume we are given a grid of three energy bins, defined
        between 1 and 3, 3 and 7, and 7 and 13 keV. Then:
        The "grid_lower_bounds" array is [1,3,7];
        the "grid_upper_bounds" array is [3,7,13]; 
        the "grid_range" array returned by this method is [1,3,7,13].  
        
        Parameters:
        ----------
        grid_lower_bounds: np.array (float)
            A one-dimensional array of length n, containing the lower bounds of
            the grid. 
            
        grid_upper_bounds: np.array(float)
            A one-dimensional array of length n, containing the upper bounds of 
            the grd. 
        
        Returns:
        ----------
        grid_range: np.array(float)
            A one-dimensional array of length n+1, containing the full extent 
            of the grid.   
        """    
 
        if (len(grid_lower_bounds) !=  len(grid_upper_bounds)):
            raise TypeError("Lower and upper grid bound arrays have different size")
            
        if (np.allclose(grid_lower_bounds[1:len(grid_lower_bounds)-1],
                        grid_upper_bounds[0:len(grid_upper_bounds)-2])
                        is not True):
            raise TypeError("Lower and upper grid bounds do not match")         
        
        grid_range = grid_lower_bounds.append(grid_upper_bounds[-1]) 
        
        return grid_range
    
    def _grid_bounds_to_widths(self,grid_lower_bounds,grid_upper_bounds):
        """
        One of five similar methods to compute quantities relevant to grid 
        arrays (such as bin widths or center points).
        
        This method takes two arrays containing the lower and upper bounds of 
        the grid, and returns a combined array which contains the width of each 
        bin in the grid.
        
        As an example, assume we are given a grid of three energy bins, defined
        between 1 and 3, 3 and 7, and 7 and 13 keV. Then:
        The "grid_lower_bounds" array is [1,3,7];
        the "grid_upper_bounds" array is [3,7,13]; 
        the "grid_widths" array returned by this method is
        [3-1,7-3,13-7] = [2,4,6].  
        
        Parameters:
        ----------
        grid_lower_bounds: np.array(float) 
            A one-dimensional array of length n, containing the lower bounds of
            the grid. 
            
        grid_upper_bounds: np.array(float)
            A one-dimensional array of length n, containing the upper bounds of 
            the grd.         
        
        Returns:        
        ----------
        grid_widths: np.array(float) 
            A one-dimensional array of length n, containing the witdh of each 
            bin in the grid.
        """    

        if (len(grid_lower_bounds) != len(grid_upper_bounds)):
            raise TypeError("Lower and upper grid bound arrays have different size")
            
        if (np.allclose(grid_lower_bounds[1:len(grid_lower_bounds)-1],
                        grid_upper_bounds[0:len(grid_upper_bounds)-2])
                        is not True):
            raise TypeError("Lower and upper grid bounds do not match")    
                    
        #For convenience, the bin widths are computed starting from the midpoint 
        #of the grid itself, computed through a different class method 
        grid_widths = np.diff(self._grid_bounds_to_midpoint(grid_lower_bounds,
                                                            grid_upper_bounds))
        
        return np.diff(grid_widths)

    def _grid_bounds_to_midpoint(self,grid_lower_bounds,grid_upper_bounds):
        """
        One of five similar methods to compute quantities relevant to grid 
        arrays (such as bin widths or center points).
        
        This method takes two arrays containing the lower and upper bounds of 
        the grid, and returns a combined array which contains the mid point of 
        each bin in the grid (defined as the arithemetic average).
        
        As an example, assume we are given a grid of three energy bins, defined
        between 1 and 3, 3 and 7, and 7 and 13 keV. Then:
        The "grid_lower_bounds" array is [1,3,7];
        the "grid_upper_bounds" array is [3,7,13]; 
        the "grid_midpoint" array  returned by this method is 
        [(3+1)/2,(7+3)/2,(13+7)/2] = [2,5,10].
        
        Parameters:
        ----------
        grid_lower_bounds: np.array(float) 
            A one-dimensional array of length n, containing the lower bounds of
            the grid. 
            
        grid_upper_bounds: np.array(float)
            A one-dimensional array of length n, containing the upper bounds of 
            the grd.    
                    
        Returns:        
        ----------
        grid_midpoint: np.array(float)         
            A one-dimensional array of length n, containing the mid point 
            (defined as the geometric average) of each bin in the grid.        
        """        

        if (len(grid_lower_bounds) != len(grid_upper_bounds)):
            raise TypeError("Lower and upper grid bound arrays have different size")
            
        if (np.allclose(grid_lower_bounds[1:len(grid_lower_bounds)-1],
                        grid_upper_bounds[0:len(grid_upper_bounds)-2])
                        is not True):
            raise TypeError("Lower and upper grid bounds do not match")    
        
        grid_midpoint = 0.5*(grid_lower_bounds+grid_upper_bounds)
        
        return grid_midpoint

    def _grid_midpoint_to_widths(self,grid_midpoint,start_point):
        """
        One of five similar methods to compute quantities relevant to grid 
        arrays (such as bin widths or center points).
        
        This method takes a) one array, containing the arithemtic mean point of 
        each point in the grid and b) the lowest bound of the first point in the
        grid, and returns an array which contains the width of each bin in the 
        grid.
        
        As an example, assume we are given a grid of three energy bins, defined
        between 1 and 3, 3 and 7, and 7 and 13 keV. Then:
        The "grid_midpoint" array is [(3+1)/2,(7+3)/2,(13+7)/2] = [2,5,10];
        the "start_point" value is 1;
        the "grid_widths" array returned by this method is
        [3-1,7-3,13-7] = [2,4,6].  
        
        Parameters:
        ----------
        grid_midpoint: np.array(float)         
            A one-dimensional array of length n, containing the mid point 
            (defined as the geometric average) of each bin in the grid. 
    
        start_point: float
            The lowest bound of the first bin in the grid.
        
        Returns:
        grid_widths: np.array(float)
            A one-dimensional array of length n, containing the witdh of each 
            bin in the grid.        
        ----------
        """
        
        if (grid_midpoint[0] < start_point):
            raise ValueError("Grid starting point is lower than the first grid mid point")
        
        #The reason we need to specify a starting point is that we are trying to 
        #determine n+1 values for the grid bounds, which we then need to
        #calculate the bin widths. We need to know n+1 values of the grid to do 
        #this, but the array of grid midpoints is only of size n, so we need one 
        #less degree of freedom (which we fix by also assuming a lower bound)         
        grid_widths = np.zeros(len(grid_midpoint))
        grid_widths[0] = 2*(grid_midpoint[0]-start_point)        
        
        for i in range(1,len(grid_midpoint)):
            bound = grid_midpoint[i-1] + 0.5*grid_widths[i-1]
            grid_widths[i] = 2*(grid_midpoint[i] - bound)            
        
        return grid_widths
   
    def _grid_midpoint_to_bounds(self,grid_midpoint,start_point):
        """
        One of five similar methods to compute quantities relevant to grid 
        arrays (such as bin widths or center points).
        
        This method takes a) one array, containing the arithemtic mean point of 
        each point in the grid and b) the lowest bound of the first point in the
        grid, and returns two arrays which contain the lower and upper bounds of 
        each bin in the grid. 
        
        As an example, assume we are given a grid of three energy bins, defined
        between 1 and 3, 3 and 7, and 7 and 13 keV. Then:
        The "grid_midpoint" array is [(3+1)/2,(7+3)/2,(13+7)/2] = [2,5,10];
        the "start_point" value is 1;
        The "grid_lower_bounds" array returned by this method is [1,3,7];
        the "grid_upper_bounds" array returned by this method is [3,7,13]. 
        
        Parameters:
        ----------
        grid_midpoint: np.array(float)         
            A one-dimensional array of length n, containing the mid point 
            (defined as the geometric average) of each bin in the grid. 
    
        start_point: float
            The lowest bound of the first bin in the grid.
                    
        Returns:
         grid_lower_bounds: np.array(float) 
            A one-dimensional array of length n, containing the lower bounds of
            the grid. 
            
        grid_upper_bounds: np.array(float)
            A one-dimensional array of length n, containing the upper bounds of 
            the grd.        
        ----------
        """

        if (grid_midpoint[0] < start_point):
            raise ValueError("Grid starting point is lower than the first grid mid point")
        
        #The reason we need to specify a starting point is that we are trying to 
        #determine n+1 values for the grid bounds. We need to know n+1 values 
        #of the grid to do this, but the array of grid midpoints is only of size
        #n, so we need one less degree of freedom  (which we fix by also 
        #assuming a lower bound)           
        grid_widths = self._grid_midpoint_to_widths(grid_midpoint,start_point)
        grid_lower_bounds = np.zeros(len(grid_midpoint))
        grid_upper_bounds = np.zeros(len(grid_midpoint))
        grid_lower_bounds[0] = starting_point
        
        for i in range(1,len(grid_midpoint)):
            grid_lower_bounds[i] = grid_midpoint[i-1] + 0.5*grid_widths[i-1]
            grid_upper_bounds[i-1] = grid_lower_bounds[i]
        
        grid_upper_bounds[-1] = midpoint[-1] + 0.5*grid_widths[-1]
        
        return grid_lower_bounds, grid_upper_bounds     
 
#tbd: widths to midpoint, widths to bounds     
    def _bounds_to_chans(self,new_lo,new_hi):
        """
        This method receives a range of lower and upper bounds of an energy or
        channel grid, and returns the appropriate array indexes in which these 
        bounds are contained.  
        
        Parameters:
        ----------  
        new_lo: np.array(float)
            An array of arbitrary size, containing the lower bounds of the new 
            grid 
            
        new_hi: np.array(float)
            An array of arbitrary size, containing the lower bounds of the new 
            grid                     
        
        Returns
        ---------- 
        return_lo: np.array(float)
            An array of size len(new_low), containing the indexes of the first 
            bin in the original grid that is contained in each bin in the input 
            grid

        return_hi: np.array(float)
            An array of size len(new_hi), containing the indexes of the last 
            bin in the original grid that is contained in each bin in the input 
            grid
        """
    
        return_lo = np.zeros(len(new_lo))
        return_hi = np.zeros(len(new_hi))
        
        for i in range(len(new_lo)):
            #find the channel numbers corresponding to the start/end of each bin
            index_lo = np.digitize(new_lo[i],self.emin)
            index_hi = np.digitize(new_hi[i],self.emax)
            return_lo[i] = index_lo
            return_hi[i] = index_hi
        
        return return_lo,return_hi          
