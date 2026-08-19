import qsample.math as math
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
        self.round_p = self.log_probs([self.partitions[self.circuit.id]], [{}])
        self.round_np = self.log_probs([{}], [self.partitions[self.circuit.id]])

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

        self.round_probs = np.arange(n_rounds)*self.round_np+self.round_p

        self.fault_loc = [deepcopy(self.fault_loc) for _ in range(n_rounds)]
        self.fault_loc_all = [self.fault_loc]
        self.failing_circuits = [[self.E_0]*n_rounds]
        self.not_empty = [any(d.values()) for i, d in enumerate(self.fault_loc)]
        self.no_fault_loc = self.no_faults(self.fault_loc)
        self.not_full = [any(d.values()) for i, d in enumerate(self.no_fault_loc)]

        self.pi_E = [self.log_probs(self.fault_loc, self.no_fault_loc)]


        # Begin with a failure configuration that fails on the first round
        self.failing_times = [1]

        #print(np.log(self.p_max['q'][0])-np.log(1-self.p_max['q'][0]))

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
            q, fault_target, fault_source, proposal, no_proposal, new_circuit_all = self.propose_relocation()
            #print(f"relocation: {q}")
            move_type_ratio = 0.0

        elif u < p_reloc + p_birth:
            q, fault_target, proposal, no_proposal, new_circuit_all = self.propose_birth()
            #print(f"birth: {q}")
            move_type_ratio = np.log(p_birth) - np.log(p_death)

        else:
            q, fault_source, proposal, no_proposal, new_circuit_all = self.propose_death()
            #print(f"death: {q}")
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



    def propose_birth(self):

        if sum(self.not_full)==0:
            return tuple([None]*5)

        new_circuit=[]
        for i in range(len(self.failing_circuits[-1])):
            new_circuit.append(deepcopy(self.failing_circuits[-1][i]))

        #possible_rounds = np.arange(len(possible_locs))[possible_locs]
        #fault_round = np.random.choice(possible_rounds, p=self.round_probs[possible_rounds]/np.sum(self.round_probs[possible_rounds]))
        R_0 = [i for i, x in enumerate(self.not_full) if x]
        fault_round = random.choice(R_0)
        
        z = [
            (k, x)
            for k, values in self.no_fault_loc[fault_round].items()
            for x in values
        ]

        
        key, item = random.choice(z)

        R_1 = [i for i, x in enumerate(self.not_empty) if x]
        if fault_round not in R_1:
            R_1.append(fault_round)
        o = [
            (k, x)
            for k, values in self.fault_loc[fault_round].items()
            for x in values
            ]

        q_birth = np.log(self.p_max[key][0])+np.log(len(R_0))+np.log(len(z))
        q_death = np.log(1-self.p_max[key][0])+np.log(len(R_1))+np.log(len(o)+1)

        #print(np.log(len(R_0))-np.log(len(R_1)))

        proposal = deepcopy(self.fault_loc)
        no_proposal = deepcopy(self.no_fault_loc)
        proposal[fault_round][key].append(item)
        no_proposal[fault_round][key].remove(item)         
        
        loc_circuit = self.err_model.run(self.circuit, {key: [item]})
        for ii in range(len(loc_circuit)):
            new_circuit[fault_round][ii] = new_circuit[fault_round][ii]|loc_circuit[ii]

        return q_birth-q_death, fault_round, proposal, no_proposal, new_circuit
        

    def propose_death(self):

        if sum(self.not_empty)==0:
            return tuple([None]*5)

        new_circuit=[]
        for i in range(len(self.failing_circuits[-1])):
            new_circuit.append(deepcopy(self.failing_circuits[-1][i]))

        #possible_rounds = np.arange(len(possible_locs))[possible_locs]
        #fault_round = np.random.choice(possible_rounds, p=self.round_probs[possible_rounds]/np.sum(self.round_probs[possible_rounds]))

        R_1 = [i for i, x in enumerate(self.not_empty) if x]
        fault_round = random.choice(R_1)
        o = [
            (k, x)
            for k, values in self.fault_loc[fault_round].items()
            for x in values
            ]
        

        
        key, item = random.choice(o)

        R_0 = [i for i, x in enumerate(self.not_full) if x]
        if fault_round not in R_0:
            R_0.append(fault_round)
                        
        z = [
            (k, x)
            for k, values in self.no_fault_loc[fault_round].items()
            for x in values
        ]


        q_birth = np.log(self.p_max[key][0])+np.log(len(R_0))+np.log(len(z)+1)
        q_death = np.log(1-self.p_max[key][0])+np.log(len(R_1))+np.log(len(o))
        #print(np.log(len(R_1))-np.log(len(R_0)))

                        

        proposal = deepcopy(self.fault_loc)
        no_proposal = deepcopy(self.no_fault_loc)
        proposal[fault_round][key].remove(item)
        no_proposal[fault_round][key].append(item)         
        
        ii = item[0]
        new_circuit[fault_round]._ticks[ii] = {}


        return q_death-q_birth, fault_round, proposal, no_proposal, new_circuit

    def propose_relocation(self):

        
        if sum(self.not_empty)==0 or sum(self.not_full)==0:
            return tuple([None]*6)

        new_circuit=[]
        for i in range(len(self.failing_circuits[-1])):
            new_circuit.append(deepcopy(self.failing_circuits[-1][i]))
        proposal = deepcopy(self.fault_loc)
        no_proposal = deepcopy(self.no_fault_loc)

        # BIRTH
        #possible_rounds = np.arange(len(not_full))[not_full]
        #fault_target = np.random.choice(possible_rounds, p=self.round_probs[possible_rounds]/np.sum(self.round_probs[possible_rounds]))
        R_0 = [i for i, x in enumerate(self.not_full) if x]
        R_1 = [i for i, x in enumerate(self.not_empty) if x]
        fault_target = random.choice(R_0)
        fault_source = random.choice(R_1)

        z_t = [
            (k, x)
            for k, values in self.no_fault_loc[fault_target].items()
            for x in values
        ]

        o_t = [
            (k, x)
            for k, values in self.fault_loc[fault_target].items()
            for x in values
        ]

        key, item = random.choice(z_t)

        
        if fault_target not in R_1:
            R_1.append(fault_target)

        
        q_birth_t = np.log(len(R_0))+np.log(len(z_t))
        q_death_t = np.log(len(R_1))+np.log(len(o_t)+1)



        proposal[fault_target][key].append(item)
        no_proposal[fault_target][key].remove(item)         
        
        loc_circuit = self.err_model.run(self.circuit, {key: [item]})
        for ii in range(len(loc_circuit)):
            new_circuit[fault_target][ii] = new_circuit[fault_target][ii]|loc_circuit[ii]

        # DEATH
        #possible_rounds = np.arange(len(not_empty))[not_empty]
        #fault_source = np.random.choice(possible_rounds, p=self.round_probs[possible_rounds]/np.sum(self.round_probs[possible_rounds]))
        z_s = [
            (k, x)
            for k, values in self.no_fault_loc[fault_source].items()
            for x in values
        ]
        
        o_s = [
            (k, x)
            for k, values in self.fault_loc[fault_source].items()
            for x in values
        ]

        key, item = random.choice(o_s)

        if fault_source not in R_0:
            R_0.append(fault_source)


        q_birth_s = np.log(len(R_0))+np.log(len(z_s)+1)
        q_death_s = np.log(len(R_1))+np.log(len(o_s))
        
        proposal[fault_source][key].remove(item)
        no_proposal[fault_source][key].append(item)         
        
        ii = item[0]
        new_circuit[fault_source]._ticks[ii] = {}

        return (q_birth_t-q_death_t)+(q_death_s-q_birth_s), fault_target, fault_source, proposal, no_proposal, new_circuit
        
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
                    return round
                
                if round < n_rounds:
                    pnode = self.protocol.root
                    msmt_hist = {}
                else:
                    return False
            