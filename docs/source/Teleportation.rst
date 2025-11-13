Teleportation
============

Teleportation protocol.

.. code-block:: python
    import qsample as qs
    import time
    import matplotlib.pyplot as plt

    teleport = qs.Circuit([{"init": {0, 1, 2}},
                       {"H": {1}},
                       {"CNOT": {(1, 2)}},
                       {"CNOT": {(0, 1)}},
                       {"H": {0}},
                       {"measure": {0, 1}}])

    meas = qs.Circuit([{"measure": {2}}], noisy=False)

    def lut(syn):
        op = {0: 'I', 1: 'X', 2: 'Z', 3: 'Y'}[syn]
        return qs.Circuit([{op: {2}}], noisy=False)

    tele_proto = qs.Protocol(check_functions={'lut': lut})
    tele_proto.add_nodes_from(['tele', 'meas'], circuits=[teleport, meas])
    tele_proto.add_edge('START', 'tele', check='True')
    tele_proto.add_edge('tele', 'COR', check='lut(tele[-1])')
    tele_proto.add_edge('COR', 'meas', check='True')
    tele_proto.add_edge('meas', 'FAIL', check='meas[-1] == 1')

    tele_proto.draw(figsize=(8,5))


Image comes here

.. code-block:: python

    err_model = qs.noise.E1
    q = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]
    err_params = {'q': q}

    ss_sam = qs.SubsetSampler(protocol=tele_proto, simulator=qs.StimSimulator,  p_max={'q': 0.1}, err_model=err_model, err_params=err_params, L=3)
    ss_sam.run(1000)

    v1 = ss_sam.stats()[0]
    v2 = ss_sam.stats()[0]
    plt.plot(q, v1)
    plt.plot(q, v2)
    plt.xscale('log')
    plt.yscale('log')
