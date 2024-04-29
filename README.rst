======
nDspec
======

|Docs|

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-dimensional X-ray data modelling in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nDspec is a software package designed to model (primarly) X-ray astronomical data; in particular, it is meant to enable users to fit spectral, timing and (in the future) polarimetric data natively. 
Currently, nDspec provides the following features:

- a class to handle folding an energy-dependent model through a given X-ray instrument response matrix. We currently support existing and legacy missions (Swift/XRT, RXTE, NuSTAR, Nicer, XMM-Newton); future missions like XRISM and Athena are not supported yet.
- Classes to handle the standard spectral-timing data products (power spectra, two-dimensional cross spectra, as well as their one-dimensional projections like lag-frequency spectra). Models need to be defined by the user either in the time domain (ie, an impulse response function) or in the Fourier domain (ie, a transfer functions). Additive models can be combined by the users before computing a given Fourier product.
- a small library of phenomenological spectral-timing models, currently including a one dimensional power-law, black body and gaussian, as well as example impulse response and/or transfer functions that parametrize the time dependence of these components.

The goal of the software is to provide users with an Xspec-like package, but to move beyond simple one-dimensional spectral fitting, which typically has relied on each group utilizing their own custom software. Being an open-source modular package, unlike legacy software nDspec allows users to interface with data science libraries for model handling or fitting. 

Documentation
-------------

The software documentation is `found on readthedocs <https://ndspec.readthedocs.io/en/latest/>`_. You can also find notebooks discussing the features of each class in the /notebooks/ folder.

Installation and testing
------------------------

The current version of the software can only be installed from the repository. Unit tests make use of `py.test <https://pytest.org>`_.

Related Packages
----------------

- `STINGRAY <https://github.com/StingraySoftware/stingray>`_ is a library that allows users to produce typical spectral-timing products from observations. A typical use would consist of using Stingray to handle the data a user might be interested in (e.g. by converting event files into data products for modelling), and nDspec to handle modelling of that data.

Contributing
------------

nDspec is a fully open source software, so we welcome your contribution and feedback!
The best way to contact the developers is through the `issues`_ page - even a simple comment on what you find useful or intuitive (or vice versa) goes a long way in helping the project. 
If you would like to contribute a piece of code, you are welcome to do so either by opening an `issue`_ or submitting a `pull request`_. 

Citing
------

The initial release paper is not released yet; until then, please refer to this repository if you make use of the software in your work.

License
-------

All content © 2023 The Authors. The code is distributed under the MIT license; see `LICENSE <LICENSE>`_ for details.

.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://ndspec.readthedocs.io/en/latest/
.. _issues: https://github.com/matteolucchini1/ndspec/issues
.. _issue: https://github.com/matteolucchini1/ndspec/issues
.. _pull request: https://github.com/matteolucchini1/ndspec/pulls
