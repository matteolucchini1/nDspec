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
    if params.ndim == 1:
        f_pk = params[0]
        q = params[1]
        rms = params[2]
    elif params.ndim == 2:
        f_pk = params[:,0][:,np.newaxis]
        q = params[:,1][:,np.newaxis]
        rms = params[:,2][:,np.newaxis]
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    #ame as Uttley and Malzac 2023
    f_res = np.divide(f_pk,np.sqrt(1.0+(1.0/(4.0*np.square(q)))))
    r = np.divide(rms,np.sqrt(0.5-np.arctan(-2.0*q)/np.pi))
    lorentz_num = (1/np.pi)*2*np.multiply(np.power(r,2),np.multiply(q,f_res))
    lorentz_den = 4*np.multiply(np.square(q),np.square(np.subtract(array,f_res)))
    model = np.divide(lorentz_num,np.square(f_res)+lorentz_den)
    return np.nan_to_num(model)

def powerlaw(array, params):
    if params.ndim == 1:
        norm = params[0]
        slope = params[1]
        model = norm*np.power(array,slope)
    elif params.ndim == 2:
        norm = params[:,0]
        slope = params[:,1]
        model = norm[:,np.newaxis]*np.power(array,slope[:,np.newaxis])
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    return model

def brokenpower(array,params):
    if params.ndim == 1:
        norm = params[0]
        slope1 = params[1]
        slope2 = params[2]
        brk = params[3]
        scaled_array = np.divide(array,brk)
        num = norm*np.power(scaled_array,slope1)
        den = 1.+np.power(scaled_array,slope1-slope2)
        model = np.divide(num,den)
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        slope1 = params[:,1][:,np.newaxis]
        slope2 = params[:,2][:,np.newaxis]
        brk = params[:,3][:,np.newaxis]
        scaled_array = np.divide(array,brk)
        num = norm*np.power(scaled_array,slope1)
        den = 1.+np.power(scaled_array,slope1-slope2)
        model = np.divide(num,den)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    return model 

def gaussian(array, params):
    if params.ndim == 1:
        center = params[0]
        width = params[1]
    elif params.ndim == 2:
        center = params[:,0][:,np.newaxis]
        width = params[:,1][:,np.newaxis]
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    norm = np.multiply(np.sqrt(2.0*np.pi),width)
    shape = np.exp(-np.power((array - center)/width,2.0)/2)
    line = shape/norm 
    return line

def bbody(array, params):
    if params.ndim == 1:
        #boltzkamnn constant in kev
        norm = params[0]
        temp = params[1]
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        temp = params[:,1][:,np.newaxis]
    renorm = 8.0525*norm/np.power(temp,4.)
    planck = np.exp(array/temp)-1.
    model = renorm*np.power(array,2.)/planck
    return model 
    
def gauss_fred(array1,array2,params,return_full=False):
    times = array1
    energy = array2
    if params.ndim == 1:
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
            fred_pulse[:,i] = norm*gaussian(energy,np.array([center,sigma[i]]))*fred_profile[i]    
        line_profile = np.sum(fred_pulse,axis=1)
        pulse_profile = np.sum(fred_pulse,axis=0)
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        width = params[:,1][:,np.newaxis]
        center = params[:,2][:,np.newaxis]
        rise_t = params[:,3][:,np.newaxis]
        decay_t = params[:,4][:,np.newaxis]
        decay_w = params[:,5][:,np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            powerlaw_shape = powerlaw(times,
                                      np.concatenate([np.ones(decay_w.shape),
                                                      decay_w],axis=1))
            sigma = np.nan_to_num(width*powerlaw_shape)
            sigma[:,0] = width.T
            fred_profile = np.exp(np.nan_to_num(-rise_t/times)-\
                                  np.nan_to_num(times/decay_t))
        fred_pulse = np.zeros((params.shape[0],len(energy),len(times)))
        line_profile = np.zeros((params.shape[0],len(energy)))
        pulse_profile = np.zeros((params.shape[0],len(times)))
        for j in range(params.shape[0]):
            for i in range(len(times)):
                par = np.array([center[j,0],sigma[j,i]])
                fred_pulse[j,:,i] = norm[j,0]*gaussian(energy,par)*fred_profile[j,i]    
            line_profile[j] = np.sum(fred_pulse[j],axis=1)
            pulse_profile[j] = np.sum(fred_pulse[j],axis=0)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    if return_full is True:
        return fred_pulse, line_profile, pulse_profile
    else:
        return fred_pulse
    
def gauss_bkn(array1,array2,params,return_full=False):
    times = array1
    energy = array2
    if params.ndim == 1:
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
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        width = params[:,1][:,np.newaxis]
        center = params[:,2][:,np.newaxis]
        rise_slope = params[:,3][:,np.newaxis]
        decay_slope = params[:,4][:,np.newaxis]
        break_time = params[:,5][:,np.newaxis]
        decay_w = params[:,6][:,np.newaxis]
        powerlaw_shape = powerlaw(times,
                                  np.concatenate([np.ones(decay_w.shape),
                                                  decay_w],axis=1))
        sigma = width*powerlaw_shape
        pars = np.concatenate([np.ones(decay_slope.shape),rise_slope,decay_slope,
                               break_time],axis=1)
        bkn_profile = brokenpower(times,pars)
        brk_pulse = np.zeros((params.shape[0],len(energy),len(times)))
        line_profile = np.zeros((params.shape[0],len(energy)))
        pulse_profile = np.zeros((params.shape[0],len(times)))
        for j in range(params.shape[0]):
            for i in range(len(times)):
                par = np.array([center[j,0],sigma[j,i]])
                brk_pulse[j,:,i] = norm[j,0]*gaussian(energy,par)*bkn_profile[j,i]    
            line_profile[j] = np.sum(brk_pulse[j],axis=1)
            pulse_profile[j] = np.sum(brk_pulse[j],axis=0)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    if return_full is True:
        return brk_pulse, line_profile, pulse_profile
    else:
        return brk_pulse
    
def pivoting_pl(array1,array2,params):
    freqs = array1
    energy = array2
    if params.ndim == 1:
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
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        pl_index = params[:,1][:,np.newaxis]
        gamma_nu = params[:,2][:,np.newaxis]
        phi_0 = params[:,3][:,np.newaxis]
        phi_slope = params[:,4][:,np.newaxis]
        nu_0 = params[:,5][:,np.newaxis]
        pivoting = np.zeros((len(energy),len(freqs)),dtype=complex)
        powerlaw_shape = norm*powerlaw(energy,
                                       np.concatenate([norm,pl_index],axis=1))
        phase = phi_0*powerlaw(freqs/nu_0,
                               np.concatenate([np.ones(phi_slope.shape),
                                               phi_slope],axis=1)) 
        #the reshaping is to avoid for loops and to use matrix multiplication instead
        log_energ = np.repeat(np.log(energy)[np.newaxis,:,np.newaxis],
                              params.shape[0],axis=0)
        piv_factor = 1 - (gamma_nu*np.exp(1j*phase))[:,np.newaxis,:]*log_energ
        pivoting = piv_factor*powerlaw_shape[:,:,np.newaxis]
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")      
    return pivoting    
    
def bbody_fred(array1,array2,params,return_full=False):
    times = array1
    energy = array2 
    if params.ndim == 1:
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
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        temp = params[:,1][:,np.newaxis]
        rise_t = params[:,2][:,np.newaxis]
        decay_t = params[:,3][:,np.newaxis]
        decay_temp = params[:,4][:,np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            temp_profile = np.nan_to_num(temp*powerlaw(times,
                                                       np.concatenate([np.ones(decay_temp.shape),
                                                                       decay_temp],axis=1)))
            fred_profile = np.exp(np.nan_to_num(-rise_t/times)-\
                                  np.nan_to_num(times/decay_t))   
        fred_pulse = np.zeros((params.shape[0],len(energy),len(times)))
        model_profile = np.zeros((params.shape[0],len(energy)))
        pulse_profile = np.zeros((params.shape[0],len(times)))
        for j in range(params.shape[0]):
            for i in range(len(times)):
                par = np.array([norm[j,0],temp_profile[j,i]])
                fred_pulse[j,:,i] = norm[j,0]*gaussian(energy,par)*fred_profile[j,i]    
            model_profile[j] = np.sum(fred_pulse[j],axis=1)
            pulse_profile[j] = np.sum(fred_pulse[j],axis=0)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    if return_full is True:
        return fred_pulse, model_profile, line_profile
    else:
        return fred_pulse
    
def bbody_bkn(array1,array2,params,return_full=False):
    times = array1
    energy = array2 
    if params.ndim == 1:
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
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        temp = params[:,1][:,np.newaxis]
        rise_slope = params[:,2][:,np.newaxis]
        decay_slope = params[:,3][:,np.newaxis]
        break_time = params[:,4][:,np.newaxis]
        decay_temp = params[:,5][:,np.newaxis]
        temp_profile = temp*powerlaw(times,np.concatenate([np.ones(decay_temp.shape),
                                                           decay_temp],axis=1)) 
        pars = np.concatenate([np.ones(decay_slope.shape),rise_slope,decay_slope,
                               break_time],axis=1)
        bkn_profile = brokenpower(times,pars)
        brk_pulse = np.zeros((params.shape[0],len(energy),len(times)))
        model_profile = np.zeros((params.shape[0],len(energy)))
        pulse_profile = np.zeros((params.shape[0],len(times)))
        for j in range(params.shape[0]):
            for i in range(len(times)):
                par = np.array([norm[j,0],temp_profile[j,i]])
                brk_pulse[j,:,i] = norm[j,0]*bbody(energy,par)*bkn_profile[j,i]    
            model_profile[j] = np.sum(brk_pulse[j],axis=1)
            pulse_profile[j] = np.sum(brk_pulse[j],axis=0)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    if return_full is True:
        return brk_pulse, model_profile, pulse_profile
    else:
        return brk_pulse  
    
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
