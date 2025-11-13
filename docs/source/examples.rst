Examples
========

GHZ
---

GHZ protocol using QSample 2.0:

.. code-block:: python

    import qsample as qs
    import time
    import stim
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import re

    ghz = qs.Circuit([ {"init": {0,1,2,3,4}},
                    {"H": {0}},
                    {"CNOT": {(0,1)}},
                    {"CNOT": {(1,2)}},
                    {"CNOT": {(2,3)}},
                    {"CNOT": {(3,4)}},
                    {"CNOT": {(0,4)}},
                    {"measure": {4}}   ])

    # Define protocol for 1 round of repetition

    def logErr(msmt_list):
        return msmt_list[-1] == 1 # If True transition to FAIL

    functions = {'logErr': logErr}

    ghz1 = qs.Protocol(check_functions=functions, fault_tolerant=False)

    ghz1.add_node('ghz', circuit=ghz) # Add node with corresponding circuit
    ghz1.add_edge('START', 'ghz', check='True') # Transition START -> first circuit node always True
    ghz1.add_edge('ghz', 'FAIL', check='logErr(ghz)')

Lorem ipsum dolor sit amet.

