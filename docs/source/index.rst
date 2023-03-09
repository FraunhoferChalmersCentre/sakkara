=====================
Sakkara
=====================
Sakkara is a framework for simplifying the creation of probabilistic models in `PyMC <https://pymc.io/>`_.

Installation
--------------------
Recommended is to first install PyMC via conda.

Then, install Sakkara via pip or poetry:

``pip install sakkara``

Getting started
--------------------

Check out the example notebooks at https://github.com/FraunhoferChalmersCentre/sakkara-examples.

API
--------------------
.. toctree::
   :maxdepth: 1
   :caption: Components

   components/distribution.rst
   components/function.rst
   components/group.rst
   components/deterministic.rst
   components/likelihood.rst
   components/fixedvalue.rst
   components/series.rst
   components/model.rst
   components/math_op.rst
   components/composable.rst
   components/hierarchical.rst

.. toctree::
   :maxdepth: 1
   :caption: Miscellaneous

   miscellaneous/build.rst
   miscellaneous/data_components.rst

.. toctree::
   :maxdepth: 1
   :caption: Relation utils

   relation_utils/group.rst
   relation_utils/representation.rst
   relation_utils/groupset.rst

Indices and tables
--------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`