.. _api:

nDspec API
==========

Docstrings for every class and function in the nDspec modelling software

Operator Class
~~~~~~~~~~~~~~

.. autoclass:: ndspec.Operator.nDspecOperator
   :members:

Response Matrix Class
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.Response.ResponseMatrix
   :members:

Timing Classes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.Timing.FourierProduct
   :members:

.. autoclass:: ndspec.Timing.PowerSpectrum
   :members:
   
.. autoclass:: ndspec.Timing.CrossSpectrum
   :members:
   
SimpleFit Classes 
~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.SimpleFit.SimpleFit
   :members:

.. autoclass:: ndspec.SimpleFit.EnergyDependentFit
   :members:
   
.. autoclass:: ndspec.SimpleFit.FrequencyDependentFit
   :members:
   
Data loading utilities
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ndspec.SimpleFit.load_lc

.. autofunction:: ndspec.SimpleFit.load_pha  
   
FitPowerSpectrum Class
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.FitPowerSpectrum.FitPowerSpectrum
   :members:
   
FitTimeAvgSpectrum Class
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.FitTimeAvgSpectrum.FitTimeAvgSpectrum
   :members:
   
FitCrossSpectrum Class
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.FitCrossSpectrum.FitCrossSpectrum
   :members:
   
Emcee sampling functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ndspec.EmceeUtils.set_emcee_priors

.. autofunction:: ndspec.EmceeUtils.set_emcee_model

.. autofunction:: ndspec.EmceeUtils.set_emcee_data

.. autofunction:: ndspec.EmceeUtils.set_emcee_parameters

.. autoclass:: ndspec.EmceeUtils.priorUniform

.. autoclass:: ndspec.EmceeUtils.priorLogUniform

.. autoclass:: ndspec.EmceeUtils.priorNormal

.. autoclass:: ndspec.EmceeUtils.priorLogNormal

.. autofunction:: ndspec.EmceeUtils.log_priors

.. autofunction:: ndspec.EmceeUtils.chi_square_likelihood

.. autofunction:: ndspec.EmceeUtils.process_emcee

Model library
~~~~~~~~~~~~~

.. autofunction:: ndspec.models.lorentz

.. autofunction:: ndspec.models.cross_lorentz

.. autofunction:: ndspec.models.powerlaw

.. autofunction:: ndspec.models.brokenpower

.. autofunction:: ndspec.models.gaussian

.. autofunction:: ndspec.models.bbody

.. autofunction:: ndspec.models.varbbody

.. autofunction:: ndspec.models.gauss_fred

.. autofunction:: ndspec.models.gauss_bkn

.. autofunction:: ndspec.models.bbody_fred

.. autofunction:: ndspec.models.bbody_bkn

.. autofunction:: ndspec.models.pivoting_pl

.. autofunction:: ndspec.models.plot_2d
