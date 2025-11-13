Steane
=====

Multiple inputs with the Steane code.

STIM

.. code-block:: python
    
    eft = qs.Circuit(noisy=True)
    sz_123 = qs.Circuit(noisy=True)
    meas7 = qs.Circuit(noisy=False)

    eft.from_stim_circuit("""R 0 1 2 3 4 5 6 7
                            H 0 1 3
                            CNOT 0 4
                            CNOT 1 2
                            TICK
                            CNOT 3 5
                            TICK
                            CNOT 0 6
                            TICK
                            CNOT 3 4
                            TICK
                            CNOT 1 5
                            TICK
                            CNOT 0 2
                            TICK
                            CNOT 5 6
                            TICK
                            CNOT 4 7
                            TICK
                            CNOT 2 7
                            TICK
                            CNOT 5 7
                            M 7""")

    sz_123.from_stim_circuit("""R 8
                            CNOT 0 8
                            TICK
                            CNOT 1 8
                            TICK
                            CNOT 3 8
                            TICK
                            CNOT 6 8
                            M 8""")

    meas7.from_stim_circuit("""M 0 1 2 3 4 5 6""")

QSample

.. code-block:: python

    eft = qs.Circuit([  {"init": {0,1,2,4,3,5,6,7}},
                        {"H": {0,1,3}},
                        {"CNOT": {(0,4)}},
                        {"CNOT": {(1,2)}},
                        {"CNOT": {(3,5)}},
                        {"CNOT": {(0,6)}},
                        {"CNOT": {(3,4)}},
                        {"CNOT": {(1,5)}},
                        {"CNOT": {(0,2)}},
                        {"CNOT": {(5,6)}},
                        {"CNOT": {(4,7)}},
                        {"CNOT": {(2,7)}},
                        {"CNOT": {(5,7)}},
                        {"measure": {7}} ])

    sz_123 = qs.Circuit([   {"init": {8}},
                            {"CNOT": {(0,8)}},
                            {"CNOT": {(1,8)}},
                            {"CNOT": {(3,8)}},
                            {"CNOT": {(6,8)}},
                            {"measure": {8}}])

    meas7 = qs.Circuit([ {"measure": {0,1,2,3,4,5,6}} ], noisy=False)



QASM

.. code-block:: python

    eft = qs.Circuit().from_qasm_circuit("""OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[8];
    creg c[1];

    h q[0];
    h q[1];
    h q[3];

    cx q[0], q[4];
    cx q[1], q[2];
    cx q[3], q[5];
    cx q[0], q[6];
    cx q[3], q[4];
    cx q[1], q[5];
    cx q[0], q[2];
    cx q[5], q[6];
    cx q[4], q[7];
    cx q[2], q[7];
    cx q[5], q[7];

    measure q[7] -> c[0];""")

Qiskit

.. code-block:: python

    q = qiskit.QuantumRegister(8)
    c = qiskit.ClassicalRegister(1)
    eft = qiskit.QuantumCircuit(q, c)

    eft.h(q[0])
    eft.h(q[1])
    eft.h(q[3])


    eft.cx(q[0], q[4])
    eft.cx(q[1], q[2])
    eft.cx(q[3], q[5])
    eft.cx(q[0], q[6])
    eft.cx(q[3], q[4])
    eft.cx(q[1], q[5])
    eft.cx(q[0], q[2])
    eft.cx(q[5], q[6])
    eft.cx(q[4], q[7])
    eft.cx(q[2], q[7])
    eft.cx(q[5], q[7])
 
    eft.measure(q[7], c[0])

    eft = qs.Circuit().from_qiskit_circuit(eft)

Steane protocol

.. code-block:: python

    k1 = 0b0001111
    k2 = 0b1010101
    k3 = 0b0110011
    k12 = k1 ^ k2
    k23 = k2 ^ k3
    k13 = k1 ^ k3
    k123 = k12 ^ k3
    stabilizerGenerators = [k1, k2, k3]
    stabilizerSet = [0, k1, k2, k3, k12, k23, k13, k123]

    def hamming2(x, y):
        count, z = 0, x ^ y
        while z:
            count += 1
            z &= z - 1
        return count

    fails = []
    def logErr(out):
        global fails
        c = np.array([hamming2(out, i) for i in stabilizerSet])
        d = np.flatnonzero(c <= 1)
        e = np.array([hamming2(out ^ (0b1111111), i) for i in stabilizerSet])
        f = np.flatnonzero(e <= 1)
        if len(d) != 0:
            return False
        elif len(f) != 0:
            fails.append(out)
            return True
        if len(d) != 0 and len(f) != 0: 
            raise('-!-!-CANNOT BE TRUE-!-!-')

    def flagged_z_look_up_table_1(z):
        s = [z]

        if s == [1]:
            return True
        else: 
            return False

    functions = {"logErr": logErr, "lut": flagged_z_look_up_table_1}

    steane0 = qs.Protocol(check_functions=functions, fault_tolerant=True)

    steane0.add_nodes_from(['ENC', 'Z2', 'meas'], circuits=[eft, sz_123, meas7])
    steane0.add_node('X_COR', circuit=qs.Circuit(noisy=True).from_stim_circuit("""X 6"""))
    steane0.add_edge('START', 'ENC', check='True')
    steane0.add_edge('ENC', 'meas', check='ENC[-1]==0')
    steane0.add_edge('ENC', 'Z2', check='ENC[-1]==1')
    steane0.add_edge('Z2', 'X_COR', check='lut(Z2[-1])')
    steane0.add_edge('Z2', 'meas', check='not lut(Z2[-1])')
    steane0.add_edge('X_COR', 'meas', check='True')
    steane0.add_edge('meas', 'FAIL', check='logErr(meas[-1])')

    err_model = qs.noise.E1
    q = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]
    err_params = {'q': q}

    begin = time.time()
    stim_sam = qs.SubsetSampler(protocol=steane0, simulator=qs.StimSimulator,  p_max={'q': 0.1}, err_model=err_model, err_params=err_params, L=3)
    stim_sam.run(2000)
    end = time.time()
    stim_time = end-begin

    v2 = stim_sam.stats()[0]
    w2 = stim_sam.stats()[2]

Lorem ipsum.
