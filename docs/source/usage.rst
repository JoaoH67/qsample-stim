Usage
=====

Basic Example
-------------

Here’s a simple quantum circuit using QSample 2.0:

.. code-block:: python

    from qsample import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    result = qc.run()
    print(result)

Advanced Usage
--------------

QSample 2.0 also supports:
- Custom gate definitions
- Circuit visualization
- Exporting results to various formats

