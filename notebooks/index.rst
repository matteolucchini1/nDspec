.. neXTsPec documentation master file, created by
   sphinx-quickstart on Fri Oct 27 14:54:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to neXTsPec's documentation!
====================================

This is the intial repository for the neXTsPec Python-based X-ray astronomy modeling software. Currently, the software allows users to correctly account for the effects of X-ray instrument responses in two-dimensional models - formally these would either be spectral-timing or spectral-polarimetry models, although the second dimension beyond photon energy does not matter. Ultimately, the code will be able to convert these convolved models into typical data products (such as lag spectra or modulation curves), support using and combining models in the Xspec library and/or custom ones, as well as interfacing with existing minimization libraries for fitting. 

==================================== 

As is obvious, the documentation is currently WiP. You can find notebooks highlighting the features of each class in the /notebooks/ folder.

==================================== 

Installation and testing

The early version of the software can only be installed from the repository. Implementation of tests is pending further development.

==================================== 

.. toctree::
    response   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
