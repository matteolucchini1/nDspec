.. nDspec documentation master file, created by
   sphinx-quickstart on Fri Oct 27 14:54:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the nDspec documentation!
====================================

This is the intial repository for the nDspec Python-based X-ray astronomy modeling software. nDspec is designed to allow users to model spectral, spectral timing and (in the future) spectral polarimetry and polarimetry timing X-ray data.

Currently, the software has two functionalities. First, users can use it to fold two-dimensional models through the response matrix of modern X-ray instruments - formally these would either be spectral-timing or spectral-polarimetry models, although the second dimension beyond photon energy does not matter. The observatories/instruments explicitely supported are RXTE/PCA, Swift/XRT, XMM-Newton, NuSTAR and NICER. Second, the nDspec can compute standard Fourier products like lag spectra from time- and/or energy- dependent, user-defined models. It is possible to input models defined in both the time and Fourier domains, as well as to combine multiple components. 

Installation and testing
=========================

The early version of the software can only be installed from the repository. Unit tests utilize py-test. 

Table of contents
================= 

.. toctree::
    response   
    timing

References and tables
=====================

.. toctree::
    api 
* :ref:`search`
