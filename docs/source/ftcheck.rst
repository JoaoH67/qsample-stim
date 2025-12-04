Fault-tolerance check
=====================

Here we demonstrate how to use the fault-tolerance check function. 
The protocols used here are the Steane protocol and the GHZ protocol.

.. code-block:: python

    n_err = 1
    stim_sam = qs.FT_Check(protocol=steane0, simulator=qs.StimSimulator,  
                       p_max={'q': 0.1}, w_max=n_err, err_model=err_model, L=3)
    stim_sam.run(2000)

.. code-block:: text

    True



.. code-block:: python

    n_err = 2
    stim_sam = qs.FT_Check(protocol=steane0, simulator=qs.StimSimulator,  
                       p_max={'q': 0.1}, w_max=n_err, err_model=err_model, L=3)
    stim_sam.run(2000)

.. code-block:: text

    False

As we can see, the Steane protocol is fault-tolerant up to a single physical error. The same is not true for the GHZ protocol:

.. code-block:: python

    n_err = 1
    stim_sam = qs.FT_Check(protocol=ghz1, simulator=qs.StimSimulator,  
                       p_max={'q': 0.1}, w_max=n_err, err_model=err_model, L=3)
    stim_sam.run(2000)

.. code-block:: text

    False