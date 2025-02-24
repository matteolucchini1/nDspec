.. nDspec documentation master file, created by
   sphinx-quickstart on Fri Oct 27 14:54:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the nDspec documentation!
====================================

This is the intial repository for the nDspec Python-based X-ray astronomy modeling software. nDspec is designed to allow users to model spectral, spectral timing and (in the future) spectral polarimetry and polarimetry timing X-ray data. The software allows users to fit power-spectra as a function of Fourier frequency, time-averaged spectra as a function of photon energy (channel), as well as cross spectra (and their related products, such as lag spectra) as a function of both Fourier frequency and energy. Bayesian sampling using the common package emcee is fully supported. The software comes with a small library of phenomenological models; alternatively, users can use their own Python-based models. Currently, nDspec does not support joint fitting multiple datasets together. 

Currently, the software is built on two core functionalities which users can use in their own code, outside of the fitting environment. The ResponseMatrix class allows user to fold one- and two-dimensional models through the response matrix of modern X-ray instruments - formally these would either be spectral-timing or spectral-polarimetry models, although the second dimension beyond photon energy does not matter. The observatories/instruments explicitely supported are RXTE/PCA, Swift/XRT, XMM-Newton, NuSTAR and NICER. The PowerSpectrum and CrossSpectrum classes can compute standard Fourier products like lag spectra from time- and/or energy- dependent, user-defined models. It is possible to input models defined in both the time and Fourier domains, as well as to combine multiple components. 

Installation and testing
~~~~~~~~~~~~~~~~~~~~~~~~

The early version of the software can only be installed from the repository. Unit tests utilize py-test. 

Table of contents
~~~~~~~~~~~~~~~~~ 

Data fitting tutorials:

.. toctree::
    fit_psd
    fit_spec
    fit_cross_1d
    fit_cross_2d

nDspec core functionality:

.. toctree::
    response_basics
    response_optimization   
    timing
    numerics

References and tables
~~~~~~~~~~~~~~~~~~~~~

* :ref:`search`
.. toctree::  
    api 
