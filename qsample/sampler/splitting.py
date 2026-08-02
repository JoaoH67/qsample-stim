from .tree import Tree, Variable, Constant, Delta
import qsample.math as math
import qsample.utils as utils
from copy import deepcopy
from ..callbacks import CallbackList
from tqdm.auto import tqdm
from time import time
import random
import numpy as np

class DirectSampler:
    """Direct Monte Carlo Sampler
    
    **Attributes:**

    protocol : Protocol
        Protocol to sample from
    simulator : StabilizerSimulator or StatevectorSimulator
        Simulator used during sampling
    err_model : ErrorModel
        Error model used during sampling
    err_params : dict
        Physical error rates per faulty partition group at which plots generated.  
    partitions : dict
        Grouping of faulty circuit elements fore each circuit in protocol
    counts : np.array
        List to accumulate counts for "marked" events
    shots : np.array
        List to accumulate shots per physical error rate
    """
    
    def __init__(self, protocol, simulator, err_model, err_params=None):
        """
        **Attributes:**

        protocol : Protocol
            The protocol to sample from
        simulator : ChpSimulator or StatevectorSimulator
            The simulator used in sampling process
        err_model : ErrorModel
            Error model used in sampling process
        err_params : dict
            Physical error rates per faulty partition group at which plots generated.
        """
        self.protocol = protocol
        self.simulator = simulator
        self.err_model = err_model()
        self.err_params = self.__err_params_to_matrix(err_params)
        self.partitions = {cid: self.err_model.group(circuit) for cid, circuit in self.protocol.circuits.items()}
        self.counts = np.array([0] * self.err_params.shape[0])
        self.shots = np.array([0] * self.err_params.shape[0])
        
        
    def __err_params_to_matrix(self, err_params):
        sorted_params = [err_params[k] for k in self.err_model.groups]
        return np.array(np.broadcast_arrays(*sorted_params)).T

            
    def stats(self, idx=None):
        """Calculate sampling statistics
        
        See Eq. 1 and C12 in paper
        
        **Returns:**

        tuple
            (Logical failure rate, Wilson variance on logical failure rate)
        """
        p_L = np.array([0 if s==0 else c/s for (c,s) in zip(self.counts, self.shots)])
        v_L = math.Wilson_var(p_L, self.shots)
        if idx != None: p_L, v_L = p_L[idx], v_L[idx]
        
        return p_L, np.sqrt(v_L)
            
    def run(self, n_shots, callbacks=[]):
        """Execute n_shots of direct Monte Carlo sampling
        
        **Attributes:**
        
        n_shots : int
            Number of shots sampled in total
        callbacks : list of Callback
            Callback instances executed during sampling
        """
        self.n_shots = n_shots
        
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(sampler=self, callbacks=callbacks)
                    
        callbacks.on_sampler_begin()
                

        pnode = self.protocol.root # get protocol start node
        state = self.simulator(len(self.protocol.qubits)) # init state
        msmt_hist = {} # init measurement history
        
        while True:
            pnode, circuit = self.protocol.successor(pnode, msmt_hist)
            if circuit != None:
                if not circuit.noisy:
                    msmt = state.run(circuit)
                else:
                    no_locs = self.err_model.choose_p(self.partitions[circuit.id], [1e-30])
                    no_fault_circuit = self.err_model.run(circuit, no_locs)
                    msmt = state.run(circuit, no_fault_circuit)
                msmt = msmt if msmt==None else int(msmt,2) # convert to int for comparison in checks
                msmt_hist[pnode] = msmt_hist.get(pnode, []) + [msmt]
            else:
                break
        
        for i, p in enumerate(self.err_params):
            
            self.stop_sampling = False # Flag can be controlled in callbacks
            self.i = i
            
            error_circuits = None
            relevant_circuit = None

            for _ in tqdm(range(n_shots), desc=f"p={tuple(map('{:.2e}'.format, p))}", leave=True):
                
                self.shots[i] += 1
                callbacks.on_protocol_begin()
                pnode = self.protocol.root # get protocol start node
                state = self.simulator(len(self.protocol.qubits)) # init state
                msmt_hist = {} # init measurement history
                
                while True:
                    callbacks.on_circuit_begin()
                    pnode, circuit = self.protocol.successor(pnode, msmt_hist)
                    if circuit != None:
                        if not circuit.noisy:
                            msmt = state.run(circuit)
                        else:
                            if relevant_circuit is None:
                                relevant_circuit = circuit
                            fault_locs = self.err_model.choose_p(self.partitions[circuit.id], p)
                            fault_circuit = self.err_model.run(circuit, fault_locs)
                            msmt = state.run(circuit, fault_circuit)
                        msmt = msmt if msmt==None else int(msmt,2) # convert to int for comparison in checks
                        msmt_hist[pnode] = msmt_hist.get(pnode, []) + [msmt]
                    else:
                        if pnode != None:
                            if error_circuits is None:
                                return fault_circuit, relevant_circuit, fault_locs
                        break
                    callbacks.on_circuit_end(locals())
                callbacks.on_protocol_end()
                if self.stop_sampling: 
                    break
        
        del self.stop_sampling
        callbacks.on_sampler_end()
        return None, None, None


class SplittingSampler:
    """Class to represent subset sampler
    
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
    def __init__(self, n_shots, protocol, simulator, p_max, err_model, distance=None):
        """
        **Attributes:**

        protocol : Protocol
            The protocol to sample from
        simulator : ChpSimulator or StatevectorSimulator
            The simulator used in sampling process
        p_max : dict
            Physical error rates per faulty partition group at which we sample
        err_model : ErrorModel
            Error model used in sampling process
        err_params : dict
            Physical error rates per faulty partition group at which plots generated.
            Should be less than p_max and it should be checked that at p_max all subsets
            scale similar. Only in this region can we use the subset sampler results.
        """
        self.protocol = protocol
        self.simulator = simulator
        
        # Calculate initial failure probability P(p1) using Direct Monte Carlo sampling
        self.monte_carlo=DirectSampler(self.protocol, self.simulator, err_model, p_max)
        self.E_0, self.circuit, self.faults = self.monte_carlo.run(n_shots) # Modified .run from original function
        self.logical_p = [self.monte_carlo.stats()[0]] # P(p1)
        self.logical_error = [self.monte_carlo.stats()[1]] # epsilon

        self.err_model = err_model()
        self.p_max = self.err_params_to_matrix(p_max) # p1
        self.distance = distance
        
        self.partitions = {cid: self.err_model.group(circuit) for cid, circuit in self.protocol.circuits.items()}
        constants = {cid: math.subset_probs(circuit, self.err_model, self.p_max) for cid, circuit in protocol.circuits.items()}
        self.tree = Tree(constants)

        self.failing_circuits = []
      
    def err_params_to_matrix(self, err_params):
        sorted_params = [err_params[k] for k in self.err_model.groups]
        return np.array(np.broadcast_arrays(*sorted_params)).T
        
    def stats(self, err_params=None):
        """Calculate statistics of sample tree wrt `err_params`
        
        See Eq. 12, Eq. 17, Eq. 18 and Eq. 19 in paper
        
        **Attributes:**

        err_params : dict or None
            Parameter range wrt to which statistics are calculated
        """                    
        _constants = self.tree.constants
        prob = self.err_params if err_params == None else self.err_params_to_matrix(err_params)
        self.tree.constants = {cid: math.subset_probs(circuit, self.err_model, prob) for cid, circuit in self.protocol.circuits.items()}
        
        p_L = self.tree.subtree_sum(self.tree.root, self.tree.marked)
        delta = self.tree.delta
        var = self.tree.var(mode=1)
        var_up = self.tree.var(mode=0)
        
        self.tree.constants = _constants
        return np.broadcast_arrays(p_L, np.sqrt(var), p_L+delta, np.sqrt(var_up))
        
    def save(self, path):
        utils.save(self, path)
        
    def calculate_subset(self, circuit):
        subset = 0
        for i in list(circuit._ticks):
            if i: subset+=1
        return subset

    def calculate_error(self, p_min, t_init, seed=None):
        """Execute n_shots of subset sampling
        
        **Attributes:**
    
        """
        if seed is None:
            seed = int((time() * 1000000000) % (2**32 - 1))
        np.random.seed(seed)
        
        self.physical_p = [self.p_max[0][0]] # p1
        self.logical_p = [self.monte_carlo.stats()[0]]
        E_0_subset=self.calculate_subset(self.E_0)
        self.t_init = t_init # Initial Markov chain length
        self.actual_lengths = []

        while self.physical_p[-1]>p_min:
            w = max(self.distance/2, len(self.circuit)*self.physical_p[-1])
            self.physical_p.append(self.physical_p[-1]*(2**(-1/np.sqrt(w))))
        self.physical_p=np.array(self.physical_p)

        n_steps = len(self.physical_p)
        N = len(self.circuit)

        #self.subset_probs = np.zeros(n_steps, N)
        p_E = np.array([self.physical_p**n for n in range(N)]).T
        np_E = (np.array([(1-self.physical_p)**n for n in range(N)]).T)[:,::-1]
        self.subset_probs = p_E*np_E

        # Create first Markov chain
        self.subsets = [E_0_subset] 
        self.pi_E=self.subset_probs[0, E_0_subset] # pi_1(E0)
        self.current_errors = [self.E_0]
        
        t = t_init
        while len(self.current_errors)<t: # Build Markov chain
            self.get_next_element(0)

        self.old_errors = deepcopy(self.current_errors) # M_1 = [E0, E1, ...]
        self.old_subsets = deepcopy(self.subsets) # sizes of E's
        
        
        for j in tqdm(range(1, n_steps)):
            
            # Reinitialize Markov chain
            self.subsets = [E_0_subset]
            self.pi_E=self.subset_probs[j][E_0_subset] #pi_j(E0)
            self.current_errors = [self.E_0] # reinitialized

            t = t_init
            scaling = 2
            # Create Markov chain
            while len(self.current_errors)<t:
                while len(self.current_errors)<t:
                    self.get_next_element(j)
                # Compute sample estimates
                if j<n_steps-1:
                    samples_minus = self.g(self.subset_probs[j, self.subsets[1:]]/
                                            self.subset_probs[j-1, self.subsets[1:]])
                    samples_plus = self.g(self.subset_probs[j, self.subsets[1:]]/
                                            self.subset_probs[j+1, self.subsets[1:]])
                    
                    g_minus = np.sum(samples_minus)/t
                    g_plus = np.sum(samples_plus)/t

                    s_minus = np.sqrt(np.sum((samples_minus-g_minus)**2)/(t-1))
                    s_plus = np.sqrt(np.sum((samples_plus-g_plus)**2)/(t-1))

                    sigma = max(s_plus/g_plus, s_minus/g_minus)/np.sqrt(t)

                    g_minus_ = np.sum(samples_minus[:int(t/2)])/int(t/2)
                    g_plus_ = np.sum(samples_plus[:int(t/2)])/int(t/2)

                    delta = max(abs(g_plus-g_plus_)/g_plus, abs(g_minus-g_minus_)/g_minus)
                    if sigma+delta > 0.25/np.sqrt(n_steps):
                        t += scaling*t_init
                        scaling*=2

            self.actual_lengths.append(len(self.subsets))
            r = self.find_ratio(j)
            self.logical_p.append(self.logical_p[-1]*r)

            #E -> E'
            self.old_errors = self.current_errors
            self.old_subsets = self.subsets

    def g(self, x):
            return 1/(1+x)
       
    def find_ratio(self, j):
        
        c=1
        
        for _ in range(3):

            c*= (np.average(self.g(c*self.subset_probs[j-1, self.old_subsets[1:]]/ # pi_j-1(E')
                                        self.subset_probs[j, self.old_subsets[1:]]))/ #pi_j(E')
            np.average(self.g(1/c*self.subset_probs[j, self.subsets[1:]]/ # pi_j-1(E)
                                        self.subset_probs[j-1, self.subsets[1:]]))) # pi_j(E)
        return c
        

    def get_next_element(self, j): # Metropolis sampling step
        new_loc = self.err_model.get_next_circuit(self.partitions[self.circuit.id])
        loc_circuit = self.err_model.run(self.circuit, new_loc)
        new_circuit = deepcopy(self.current_errors[-1])
        for i in range(len(loc_circuit._ticks)):
            if loc_circuit._ticks[i]:
                qubit_loc = list(loc_circuit[i].values())
                for ii in qubit_loc:
                    if ii in list(new_circuit[i].values()):
                        index = list(new_circuit[i].values()).index(ii)
                        key = list(new_circuit[i].keys())[index]   # index you want
                        new_circuit[i].pop(key)
                        
                    else:
                        new_circuit[i] = new_circuit[i]|loc_circuit[i]

        new_subset = self.calculate_subset(new_circuit)
        pi_E_new = self.subset_probs[j, new_subset]
        q = pi_E_new/self.pi_E

        if np.random.rand(1)<q:
            if self.run(new_circuit):
                self.failing_circuits.append(new_circuit)
                self.current_errors.append(new_circuit)
                self.pi_E = pi_E_new
                self.subsets.append(new_subset)
        
    def run(self, fault_circuit):

        pnode = self.protocol.root # get protocol start node
        state = self.simulator(max(self.protocol.qubits)+1) # init state
        msmt_hist = {} # init measurement history
        while True:
            pnode, circuit = self.protocol.successor(pnode, msmt_hist)
            if circuit != None:
                if not circuit.noisy:
                    msmt = state.run(circuit)
                else:
                    msmt = state.run(circuit, fault_circuit)

                msmt = msmt if msmt==None else int(msmt,2) # convert to int for comparison in checks
                msmt_hist[pnode] = msmt_hist.get(pnode, []) + [msmt]
            else:
                return (pnode != None)
            


class LifetimeSampler:
    
    def __init__(self, n_shots, protocol, simulator, p_max, err_model):

        self.protocol = protocol
        self.simulator = simulator
        self.p_max = p_max
        self.err_model = err_model()
        
        self.monte_carlo=DirectSampler(self.protocol, self.simulator, err_model, p_max)
        self.E_0, self.circuit, self.fault_loc = self.monte_carlo.run(n_shots) # Modified .run from original function


        self.err_groups = self.err_model.group(self.circuit)
        self.max_faults = 0
        for key in self.err_groups.keys():
            self.max_faults = len(self.err_groups[key])

        
        self.partitions = {cid: self.err_model.group(circuit) for cid, circuit in self.protocol.circuits.items()}
    
        
    def log_probs(self, faults, no_faults):

        arr = np.array([
            self.p_max[k][0]
            for d in faults
            for k, items in d.items()
            for _ in items
        ])

        no_arr = np.array([1-self.p_max[k][0]
                for d in no_faults
                for k, items in d.items()
                for _ in items
            ])

        return np.sum(np.log(arr))+np.sum(np.log(no_arr))
    
    
    def no_faults(self, faults):
        return [
            {
                k: [x for x in self.err_groups.get(k, []) if x not in d.get(k, [])]
                for k in self.err_groups
            }
            for d in faults
        ]

    def calculate_lifetime(self, 
                           n_rounds=50, 
                           t_init=1000, 
                           p=0.3, 
                           burn_in=500, 
                           thin=10, 
                           move_probs=(0.60, 0.20, 0.20), 
                           seed=None):
        """
        
        **Attributes:**
    
        """
        if seed is None:
            seed = int((time() * 1000000000) % (2**32 - 1))
        np.random.seed(seed)

        self.t_init = t_init
        self.physical_p = p 
        self.burn_in = burn_in
        self.thin = thin
        self.move_probs = move_probs

        self.fault_loc = [deepcopy(self.fault_loc) for _ in range(n_rounds)]
        self.fault_loc_all = [self.fault_loc]
        self.failing_circuits = [[self.E_0]*n_rounds]
        self.not_empty = [any(d.values()) for i, d in enumerate(self.fault_loc)]
        self.no_fault_loc = self.no_faults(self.fault_loc)
        self.not_full = [any(d.values()) for i, d in enumerate(self.no_fault_loc)]

        self.pi_E = [self.log_probs(self.fault_loc, self.no_fault_loc)]


        # Begin with a failure configuration that fails on the first round
        self.failing_times = [1]

        accepted_burn = 0
        for _ in tqdm(range(self.burn_in)):
            accepted = self.get_next_element(n_rounds=n_rounds, move_probs=move_probs)
            accepted_burn += int(accepted)

        accepted_main = 0    
        steps_main = 0
        for _ in tqdm(range(self.t_init)):
            for _ in range(self.thin):
                accepted = self.get_next_element(n_rounds=n_rounds, move_probs=move_probs)
                accepted_main += int(accepted)
                steps_main += 1


        self.failing_times = np.array(self.failing_times)
        #self.failing_subsets_all = np.array(self.failing_subsets)

        #self.failing_weights = np.zeros(self.t_init)
        #for i in range(self.t_init):
         #   self.failing_weights[i] = np.sum(self.failing_subsets_all[i,:self.failing_times[i]])

        diagnostics = {
                "accept_burn": accepted_burn / max(1, self.burn_in),
                "accept_main": accepted_main / max(1, steps_main),
                #"final_weight": self.failing_weights,
                #"final_first_T": self.failing_times,
                "average_first_T": np.average(self.failing_times[self.burn_in:])
            }
        
        return diagnostics
        


    

    
    def get_next_element(self, 
                         n_rounds=50, 
                         move_probs=(0.60, 0.20, 0.20)): # Metropolis sampling step
        
        u = np.random.rand(1)


        p_reloc, p_birth, p_death = move_probs
        fault_target = None
        fault_source = None

        if u < p_reloc:
            q, fault_target, fault_source, proposal, no_proposal, new_circuit_all = self.propose_relocation(self.not_empty, self.not_full)
            move_type_ratio = 0.0

        elif u < p_reloc + p_birth:
            q, fault_target, proposal, no_proposal, new_circuit_all = self.propose_birth(self.not_full)
            move_type_ratio = np.log(p_birth) - np.log(p_death)

        else:
            q, fault_source, proposal, no_proposal, new_circuit_all = self.propose_death(self.not_empty)
            move_type_ratio = np.log(p_death) - np.log(p_birth)

        if new_circuit_all is not None:
            

            if np.random.rand(1)<np.exp(q+move_type_ratio): # Only run circuit if it gets accepted in the first place

                fail_round =  self.run(new_circuit_all, n_rounds)
                if fail_round: # If FAIL 
                    self.failing_times.append(fail_round)
                    self.failing_circuits.append(new_circuit_all)

                    self.fault_loc = proposal
                    self.fault_loc_all.append(self.fault_loc)
                    self.no_fault_loc = no_proposal

                    if fault_target is not None:
                        self.not_empty[fault_target] = True
                        self.not_full[fault_target] = any(self.no_fault_loc[fault_target].values())
                    if fault_source is not None:
                        self.not_empty[fault_source] = any(self.fault_loc[fault_source].values())
                        self.not_full[fault_source] = True

                    self.pi_E.append(self.pi_E[-1]+q)

                    return True
                else:
                    return False 
            else:
                return False
        else:
            return False



    def propose_birth(self, possible_locs):

        if sum(possible_locs)==0:
            return tuple([None]*5)

        new_circuit=[]
        for i in range(len(self.failing_circuits[-1])):
            new_circuit.append(deepcopy(self.failing_circuits[-1][i]))


        fault_round = random.choice([i for i, x in enumerate(possible_locs) if x])

        key, item = random.choice([
            (k, x)
            for k, values in self.no_fault_loc[fault_round].items()
            for x in values
        ])

        q = np.log(self.p_max[key][0])-np.log(1-self.p_max[key][0])
                        

        proposal = deepcopy(self.fault_loc)
        no_proposal = deepcopy(self.no_fault_loc)
        proposal[fault_round][key].append(item)
        no_proposal[fault_round][key].remove(item)         
        
        loc_circuit = self.err_model.run(self.circuit, {key: [item]})
        for ii in range(len(loc_circuit)):
            new_circuit[fault_round][ii] = new_circuit[fault_round][ii]|loc_circuit[ii]

        return q, fault_round, proposal, no_proposal, new_circuit
        

    def propose_death(self, possible_locs):

        if sum(possible_locs)==0:
            return tuple([None]*5)

        new_circuit=[]
        for i in range(len(self.failing_circuits[-1])):
            new_circuit.append(deepcopy(self.failing_circuits[-1][i]))


        fault_round = random.choice([i for i, x in enumerate(possible_locs) if x])

        key, item = random.choice([
            (k, x)
            for k, values in self.fault_loc[fault_round].items()
            for x in values
        ])

        q = -np.log(self.p_max[key][0])+np.log(1-self.p_max[key][0])
                        

        proposal = deepcopy(self.fault_loc)
        no_proposal = deepcopy(self.no_fault_loc)
        proposal[fault_round][key].remove(item)
        no_proposal[fault_round][key].append(item)         
        
        ii = item[0]
        new_circuit[fault_round]._ticks[ii] = {}


        return q, fault_round, proposal, no_proposal, new_circuit

    def propose_relocation(self, not_empty, not_full):

        
        if sum(not_empty)==0 or sum(not_full)==0:
            return tuple([None]*6)

        new_circuit=[]
        for i in range(len(self.failing_circuits[-1])):
            new_circuit.append(deepcopy(self.failing_circuits[-1][i]))
        proposal = deepcopy(self.fault_loc)
        no_proposal = deepcopy(self.no_fault_loc)

        # BIRTH
        fault_target = random.choice([i for i, x in enumerate(not_full) if x])


        key, item = random.choice([
            (k, x)
            for k, values in self.no_fault_loc[fault_target].items()
            for x in values
        ])

        q = np.log(self.p_max[key][0])-np.log(1-self.p_max[key][0])
        proposal[fault_target][key].append(item)
        no_proposal[fault_target][key].remove(item)         
        
        loc_circuit = self.err_model.run(self.circuit, {key: [item]})
        for ii in range(len(loc_circuit)):
            new_circuit[fault_target][ii] = new_circuit[fault_target][ii]|loc_circuit[ii]

        # DEATH
        fault_source = random.choice([i for i, x in enumerate(not_empty) if x])


        key, item = random.choice([
            (k, x)
            for k, values in self.fault_loc[fault_source].items()
            for x in values
        ])

        q += (-np.log(self.p_max[key][0])+np.log(1-self.p_max[key][0]))
        proposal[fault_source][key].remove(item)
        no_proposal[fault_source][key].append(item)         
        
        ii = item[0]
        new_circuit[fault_source]._ticks[ii] = {}

        return q, fault_target, fault_source, proposal, no_proposal, new_circuit
        
    def run(self, fault_circuits, n_rounds):

        round = 0
        pnode = self.protocol.root # get protocol start node
        state = self.simulator(max(self.protocol.qubits)+1) # init state
        msmt_hist = {} # init measurement history

        while True:
            pnode, circuit = self.protocol.successor(pnode, msmt_hist)
            if circuit != None:
                if not circuit.noisy:
                    msmt = state.run(circuit)
                else:
                    msmt = state.run(circuit, fault_circuits[round])
                msmt = msmt if msmt==None else int(msmt,2) # convert to int for comparison in checks
                msmt_hist[pnode] = msmt_hist.get(pnode, []) + [msmt]
            else:
                round += 1
                if (pnode != None):
                    return deepcopy(round)
                
                if round < n_rounds:
                    pnode = self.protocol.root
                    msmt_hist = {}
                else:
                    return False
            