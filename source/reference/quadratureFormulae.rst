.. _quadratureFormulae:

Quadrature formulae
===================

.. todo:: Write only freefem code, not math equation, refere to doc / fintie element

.. math::
   \newcommand{\boldx}{\mathbf{x}}
   \newcommand{\boldxi}{\boldsymbol{\xi}}

The quadrature formula is like the following:

.. math::
   \int_{D}{f(\boldx)} \approx \sum_{\ell=1}^{L}{\omega_\ell f(\boldxi_\ell)}

.. _quadratureFormulaeInt1d:

int1d
-----

Quadrature formula on an edge.

Notations
~~~~~~~~~

:math:`|D|` is the measure of the edge :math:`D`.

For a shake of simplicity, we denote:

.. math::
   f(\boldx) = g(t)

with :math:`0\leq t\leq 1`; :math:`\boldx=(1-t)\boldx_0+t\boldx_1`.

qf1pE
~~~~~

.. code:: freefem

   int1d(Th, qfe=qf1pE)( ... )

or

.. code:: freefem

   int1d(Th, qfe=qforder=2)( ... )

This quadrature formula is exact on :math:`\mathbb{P}_1`.

.. math::
   \int_{D}{f(\boldx)} \approx |D|g\left(\frac{1}{2}\right)

qf2pE
~~~~~

.. code:: freefem

   int1d(Th, qfe=qf2pE)( ... )

or

.. code:: freefem

   int1d(Th, qfe=qforder=3)( ... )

This quadrature formula is exact on :math:`\mathbb{P}_3`.

.. math::
   \int_{D}{f(\boldx)} \approx \frac{|D|}{2}\left(
         g\left( \frac{1+\sqrt{1/3}}{2} \right)
       + g\left( \frac{1-\sqrt{1/3}}{2} \right)
   \right)

qf3pE
~~~~~

.. code:: freefem

   int1d(Th, qfe=qf3pE)( ... )

or

.. code:: freefem

   int1d(Th, qfe=qforder=6)( ... )

This quadrature formula is exact on :math:`\mathbb{P}_5`.

.. math::


   \int_{D}{f(\boldx)} \approx \frac{|D|}{18}\left(
         5g\left( \frac{1+\sqrt{3/5}}{2} \right)
       + 8g\left( \frac{1}{2} \right)
       + 5g\left( \frac{1-\sqrt{3/5}}{2} \right)
   \right)

qf4pE
~~~~~

.. code:: freefem

   int1d(Th, qfe=qf4pE)( ... )

or

.. code:: freefem

   int1d(Th, qfe=qforder=8)( ... )

This quadrature formula is exact on :math:`\mathbb{P}_7`.

.. math::
   \int_{D}{f(\boldx)} \approx \frac{|D|}{72}\left(
         (18-\sqrt{30})g\left( \frac{1-\frac{\sqrt{525+70\sqrt{30}}}{35}}{2} \right)
       + (18-\sqrt{30})g\left( \frac{1+\frac{\sqrt{525+70\sqrt{30}}}{35}}{2} \right)
       + (18+\sqrt{30})g\left( \frac{1-\frac{\sqrt{525-70\sqrt{30}}}{35}}{2} \right)
       + (18+\sqrt{30})g\left( \frac{1+\frac{\sqrt{525-70\sqrt{30}}}{35}}{2} \right)
   \right)

qf5pE
~~~~~

.. code:: freefem

   int1d(Th, qfe=qf5pE)( ... )

or

.. code:: freefem

   int1d(Th, qfe=qforder=10)( ... )

This quadrature formula is exact on :math:`\mathbb{P}_9`.

.. math::
   \int_{D}{f(\boldx)} \approx |D|\left(
         \frac{(332-13\sqrt{70})}{1800}g\left( \frac{1-\frac{\sqrt{245+14\sqrt{70}}}{21}}{2} \right)
       + \frac{(332-13\sqrt{70})}{1800}g\left( \frac{1+\frac{\sqrt{245+14\sqrt{70}}}{21}}{2} \right)
       + \frac{64}{225}g\left( \frac{1}{2} \right)
       + \frac{(332+13\sqrt{70})}{1800}g\left( \frac{1-\frac{\sqrt{245-14\sqrt{70}}}{21}}{2} \right)
       + \frac{(332+13\sqrt{70})}{1800}g\left( \frac{1+\frac{\sqrt{245-14\sqrt{70}}}{21}}{2} \right)
   \right)

qf1pElump
~~~~~~~~~

.. code:: freefem

   int1d(Th, qfe=qf1pElump)( ... )

This quadrature formula is exact on :math:`\mathbb{P}_2`.

.. math::
   \int_{D}{f(\boldx)} \approx \frac{|D|}{2}\left(
         g\left( 0 \right)
       + g\left( 1 \right)
   \right)

.. _quadratureFormulaeInt2d:

int2d
-----

qf1pT
~~~~~

.. todo:: todo

qf2pT
~~~~~

.. todo:: todo

qf5pT
~~~~~

.. todo:: todo

qf1pTlump
~~~~~~~~~

.. todo:: todo

qf2pT4P1
~~~~~~~~

.. todo:: todo

qf7pT
~~~~~

.. todo:: todo

qf9pT
~~~~~

.. todo:: todo

.. _quadratureFormulaeInt3d:

int3d
-----

qfV1
~~~~

.. todo:: todo

qfV2
~~~~

.. todo:: todo

qfV5
~~~~

.. todo:: todo

qfV1lump
~~~~~~~~

.. todo:: todo
