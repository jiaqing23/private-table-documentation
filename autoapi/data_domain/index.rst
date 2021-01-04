:mod:`data_domain`
==================

.. py:module:: data_domain


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   data_domain.DataDomain
   data_domain.RealDataDomain
   data_domain.CategoricalDataDomain



.. py:class:: DataDomain

   Bases: :class:`abc.ABC`

   Representing the set of all possible values for a column in the private table.

   There are two sub-types:
       - RealDataDomain: a range of real value [left, right]
       - CategoricalDataDomain: a list of discrete values [a, b, c, d]

   .. method:: contains(self, value: Union[float, Any])
      :abstractmethod:



.. py:class:: RealDataDomain(lower_bound: float, upper_bound: float)

   Bases: :class:`data_domain.DataDomain`

   A range of real values: [left, right].

   .. method:: contains(self, value: Union[float, Any]) -> bool


   .. method:: length(self) -> float


   .. method:: __repr__(self)

      Return repr(self).



.. py:class:: CategoricalDataDomain(values: Iterable[Any])

   Bases: :class:`data_domain.DataDomain`

   A list of discrete values.

   .. method:: contains(self, value: Union[float, Any]) -> bool


   .. method:: __repr__(self)

      Return repr(self).



