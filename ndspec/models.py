import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
fi = 22
plt.rcParams.update({'font.size': fi-5})

colorscale = pl.cm.PuRd(np.linspace(0.,1.,5))

def lorentz(array, params):
    f_pk = params[0]
    q = params[1]
    rms = params[2]
    #ame as Uttley and Malzac 2023
    f_res = np.divide(f_pk,np.sqrt(1.0+(1.0/(4.0*np.square(q)))))
    r = np.divide(rms,np.sqrt(0.5-np.arctan(-2.0*q)/np.pi))
    lorentz_num = (1/np.pi)*2*np.multiply(np.power(r,2),np.multiply(q,f_res))
    lorentz_den = 4*np.multiply(np.square(q),np.square(np.subtract(array,f_res)))
    model = np.divide(lorentz_num,np.square(f_res)+lorentz_den)
    return np.transpose(np.nan_to_num(model))

def powerlaw(array, params):
    norm = params[0]
    slope = params[1]
    model = norm*np.power(array,slope)
    return model

def brokenpower(array,params):
    norm = params[0]
    slope1 = params[1]
    slope2 = params[2]
    brk = params[3]
    scaled_array = np.divide(array,brk)
    num = norm*np.power(scaled_array,slope1)
    den = 1.+np.power(scaled_array,slope1-slope2)
    model = np.divide(num,den)
    return model 

def gaussian(array, params):
    center = params[0]
    width = params[1]
    norm = np.multiply(np.sqrt(2.0*np.pi),width)
    shape = np.exp(-np.power((array - center)/width,2.0)/2)
    line = shape/norm 
    return line

def bbody(array, params):
    #boltzkamnn constant in kev
    norm = params[0]
    temp = params[1]
    renorm = 8.0525*norm/np.power(temp,4.)
    planck = np.exp(array/temp)-1.
    model = renorm*np.power(array,2.)/planck
    return model 
    
def gauss_fred(array1,array2,params):
    times = array1
    energy = array2
    norm = params[0]
    width = params[1]
    center = params[2]
    rise_t = params[3]
    decay_t = params[4]
    decay_w = params[5]
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = np.nan_to_num(width*powerlaw(times,np.array([1.,decay_w])))
        sigma[0] = width
        fred_profile = np.exp(np.nan_to_num(-rise_t/times)-\
                              np.nan_to_num(times/decay_t))
    fred_pulse = np.zeros((len(energy),len(times)))
    line_profile = np.zeros(len(energy))
    pulse_profile = np.zeros(len(times))
    for i in range(len(times)):
        fred_pulse[:,i] = gaussian(energy,np.array([center,sigma[i]]))*fred_profile[i]    
    line_profile = np.sum(fred_pulse,axis=1)
    pulse_profile = np.sum(fred_pulse,axis=0)
    return fred_pulse, line_profile, pulse_profile
    
def gauss_bkn(array1,array2,params):
    times = array1
    energy = array2
    norm = params[0]
    width = params[1]
    center = params[2]
    rise_slope = params[3]
    decay_slope = params[4]
    break_time = params[5]
    decay_w = params[6]
    sigma = width*powerlaw(times,np.array([1.,decay_w]))
    bkn_profile = brokenpower(times,np.array([1.,rise_slope,decay_slope,break_time]))
    brk_pulse = np.zeros((len(energy),len(times)))
    line_profile = np.zeros(len(energy))
    pulse_profile = np.zeros(len(times))
    for i in range(len(times)):
        brk_pulse[:,i] = norm*gaussian(energy,np.array([center,sigma[i]]))*bkn_profile[i]    
    line_profile = np.sum(brk_pulse,axis=1)
    pulse_profile = np.sum(brk_pulse,axis=0)
    return brk_pulse, line_profile, pulse_profile
    
def pivoting_pl(array1,array2,params):
    freqs = array1
    energy = array2
    norm = params[0]
    pl_index = params[1]
    gamma_nu = params[2]
    phi_0 = params[3]
    phi_slope = params[4]
    nu_0 = params[5]
    pivoting = np.zeros((len(energy),len(freqs)),dtype=complex)
    powerlaw_shape = norm*powerlaw(energy,np.array([norm,pl_index]))
    phase = phi_0*powerlaw(freqs/nu_0,np.array([1.,phi_slope])) 
    #the reshaping is to avoid for loops and to use matrix multiplication instead
    piv_factor = 1 - gamma_nu*np.exp(1j*phase).reshape((1,len(freqs)))*np.log(energy).reshape((len(energy),1))
    pivoting = piv_factor*powerlaw_shape.reshape(len(energy),1)
    return pivoting    
    
def bbody_fred(array1,array2,params):
    times = array1
    energy = array2 
    norm = params[0]
    temp = params[1]
    rise_t = params[2]
    decay_t = params[3]
    decay_temp = params[4]
    with np.errstate(divide='ignore', invalid='ignore'):
        temp_profile = np.nan_to_num(temp*powerlaw(times,np.array([1.,decay_temp])))
        fred_profile = np.exp(np.nan_to_num(-rise_t/times)-\
                              np.nan_to_num(times/decay_t))   
    fred_pulse = np.zeros((len(energy),len(times)))
    model_profile = np.zeros(len(energy))
    pulse_profile = np.zeros(len(times))
    for i in range(len(times)):
        fred_pulse[:,i] = bbody(energy,np.array([norm,temp_profile[i]]))*fred_profile[i]
    model_profile = np.sum(fred_pulse,axis=1)
    pulse_profile = np.sum(fred_pulse,axis=0)
    return fred_pulse, model_profile, pulse_profile  
    
def bbody_bkn(array1,array2,params):
    times = array1
    energy = array2 
    norm = params[0]
    temp = params[1]
    rise_slope = params[2]
    decay_slope = params[3]
    break_time = params[4]
    decay_temp = params[5]
    temp_profile = temp*powerlaw(times,np.array([1.,decay_temp]))
    bkn_profile = brokenpower(times,np.array([1.,rise_slope,decay_slope,break_time]))
    brk_pulse = np.zeros((len(energy),len(times)))
    model_profile = np.zeros(len(energy))
    pulse_profile = np.zeros(len(times))
    for i in range(len(times)):
        brk_pulse[:,i] = bbody(energy,np.array([norm,temp_profile[i]]))*bkn_profile[i]
    model_profile = np.sum(brk_pulse,axis=1)
    pulse_profile = np.sum(brk_pulse,axis=0)
    return brk_pulse, model_profile, pulse_profile   
    
def plot_2d(xaxis,yaxis,impulse_2d,impulse_x,impulse_y,
            xlim=[0.,400.],ylim=[0.1,10.5],xlog=False,ylog=False,
            return_plot=False,normalize_en=True):
    fig = plt.figure(figsize=(9.,7.5))

    gs = gridspec.GridSpec(200,200)
    gs.update(wspace=0,hspace=0)
    ax = plt.subplot(gs[:-50,:-50])
    side = plt.subplot(gs[:-50,-50:200])
    below = plt.subplot(gs[-50:200,:-50])

    if normalize_en is True:
        impulse_2d = yaxis.reshape(len(yaxis),1)**2*impulse_2d
        impulse_y = impulse_y*yaxis**2

    c = ax.pcolormesh(xaxis,yaxis,impulse_2d,cmap="PuRd",
                  shading='auto',linewidth=0,rasterized=True)
    ax.set_xticklabels([])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_visible(False)
    ax.set_ylabel("Energy (keV)",fontsize=18)

    below.semilogy(xaxis,impulse_x,linewidth=2.5,color=colorscale[3])
    below.set_xlabel("Time ($\\rm{R_g}/c$)",fontsize=18)
    below.set_ylabel("Response",fontsize=18)
    below.set_xlim(xlim)
    below.set_ylim([1e-4*(max(impulse_x)),2.5*max(impulse_x)])

    side.step(impulse_y,yaxis,linewidth=2.5,color=colorscale[3],where='mid')
    side.invert_xaxis()
    side.yaxis.tick_right()
    side.yaxis.set_label_position('right')
    side.yaxis.set_ticks_position('both')
    side.set_xlabel("Spectrum \n (arb. units)",fontsize=18)
    side.yaxis.set_visible(False)
    side.set_ylim(ylim)
    fig.colorbar(c, ax=side)
    
    if ylog is True:
        ax.set_yscale("log",base=10)
        side.set_yscale("log",base=10)

    if xlog is True:
        ax.set_xscale("log",base=10)
        side.set_xscale("log",base=10)

    plt.show()
    if return_plot is True:
        return fig 
    else:
        return       
