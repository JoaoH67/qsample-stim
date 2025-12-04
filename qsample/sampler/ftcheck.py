from .tree import Tree, Variable
import qsample.math as math
import qsample.utils as utils

from tqdm.auto import tqdm

import numpy as np

class FT_Check:
    """Class to represent the fault tolerance check
    
    **Attributes:**

    protocol : Protocol
        Protocol to sample from
    simulator : StabilizerSimulator or StatevectorSimulator
        Simulator used during sampling
    err_model : ErrorModel
        Error model used during sampling
    err_params : dict
        Physical error rates per faulty partition group at which plots generated.
    p_max : dict
        Error probabilities per faulty partition member (one float per member)
    partitions : dict
        Grouping of faulty circuit elements fore each circuit in protocol
    tree : CountTree
        Tree data structure to keep track of sampled events
    """
    def __init__(self, protocol, simulator, p_max, w_max, err_model, L=None):
        """
        **Attributes:**

        protocol : Protocol
            The protocol to sample from
        simulator : ChpSimulator or StatevectorSimulator
            The simulator used in sampling process
        p_max : dict
            Physical error rates per faulty partition group at which we sample
        w_max : int
            Maximum number of physical errors to be probed
        err_model : ErrorModel
            Error model used in sampling process
        """
        self.protocol = protocol
        self.simulator = simulator
        self.err_model = err_model()
        self.p_max = self.err_params_to_matrix(p_max)
        self.w_max = w_max
        
        self.partitions = {cid: self.err_model.group(circuit) for cid, circuit in self.protocol.circuits.items()}
        constants = {cid: math.subset_probs(circuit, self.err_model, self.p_max) for cid, circuit in protocol.circuits.items()}
        self.tree = Tree(constants, L)

    def err_params_to_matrix(self, err_params):
        sorted_params = [err_params[k] for k in self.err_model.groups]
        return np.array(np.broadcast_arrays(*sorted_params)).T
        
    def save(self, path):
        utils.save(self, path)
        
    def _choose_subset(self, tnode, circuit):
        """Choose a subset for `circuit`, based on current `tnode`
        
        Choice is based on subset occurence probability (Aws).
        
        See App. C3a in paper
        
        **Attributes:**

        tnode : Variable
            Current tree node we want to sample from
        circuit : Circuit
            Current circuit associated with tree node
        
        **Returns:**

        tuple
            Next subset to choose for `tnode`
        """
        subsets, Aws = zip(*self.tree.constants[circuit.id].items())
        choices = min(len(subsets), self.max_weight+1)
        Aws_normalized = Aws[:choices]/sum(Aws[:choices])
        return subsets[ np.random.choice(np.arange(choices), p=Aws_normalized) ]

        
    def run(self, n_shots):
        """Execute n_shots of subset sampling
        
        **Attributes:**
        
        n_shots : int
            Number of shots sampled in total
        callbacks : list of Callback
            Callback instances executed during sampling

        **Returns:**

        fault_tolerant: bool
            Returns whether protocol is fault tolerant up to w_max faults
        """
        self.n_shots = n_shots
        self.stop_sampling = False # Flag can be controlled in callbacks
        
        for _ in tqdm(range(n_shots), desc=f"p={tuple(map('{:.2e}'.format, self.p_max))}"):
            pnode = self.protocol.root # get protocol start node
            state = self.simulator(max(self.protocol.qubits)+1) # init state
            msmt_hist = {} # init measurement history
            tnode = None # init tree node
            self.max_weight = self.w_max
            

            while True:
                pnode, circuit = self.protocol.successor(pnode, msmt_hist)
                if pnode == 'FAIL':
                    return False
                else:
                    pass
                tnode = self.tree.add(name=pnode, parent=tnode, node_type=Variable)
                tnode.count += 1
                
                if circuit != None:
                    tnode.circuit_id = circuit.id
                    
                    if not circuit.noisy:
                        msmt = state.run(circuit)
                    else:
                        subset = self._choose_subset(tnode, circuit)
                        self.max_weight-=subset[0]
                        fault_locs = self.err_model.choose_w(self.partitions[circuit.id], subset)
                        fault_circuit = self.err_model.run(circuit, fault_locs)
                        msmt = state.run(circuit, fault_circuit)
                        
                    msmt = msmt if msmt==None else int(msmt,2) # convert to int for comparison in checks
                    msmt_hist[pnode] = msmt_hist.get(pnode, []) + [msmt]
                else:
                    break

            if self.stop_sampling:
                break
        del self.stop_sampling
        return True