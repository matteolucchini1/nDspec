========
NEXTSPEC PROTOTYPING
========

This is the intial repository for the neXTsPec Python-based X-ray astronomy modeling software. Currently, the software allows users to correctly account for the effects of X-ray instrument responses in two-dimensional models - formally these would either be spectral-timing or spectral-polarimetry models, although the second dimension beyond photon energy does not matter. Ultimately, the code will be able to convert these convolved models into typical data products (such as lag spectra or modulation curves), support using and combining models in the Xspec library and/or custom ones, as well as interfacing with existing minimization libraries for fitting. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The documentation is currently WiP. You can find notebooks highlighting the features of each class in the /notebooks/ folder.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Installation and testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The early version of the software can only be installed from the repository. Implementation of tests is pending further development.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Citing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial release paper is not released yet; until then, please refer to this repository if you make use of the software in your work.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
License
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All content © 2023 The Authors. The code is distributed under the MIT license; see `LICENSE <LICENSE>`_ for details.
