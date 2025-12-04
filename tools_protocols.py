import stim

import numpy as np

from ldpc import BpOsdDecoder

from ast import literal_eval
from collections import defaultdict
from tqdm import tqdm
from line_profiler import profile
from functools import reduce

from tools_noise import noisy_circuit
from tools_io import list_of_dicts_to_dict_of_arrays
from tools_stim import choose_indices_of_pauli_string
from tools_decoder import syndrome_of_pauli_string, load_lookup_table, save_lookup_table
from tools_experiment import Experiment, coloration_schedule
from tools_codes import LCSCode
from tools_circuits import PauliAnalyzer, CircuitAnalyzer

        


@profile
def construct_flag_error_set(hz, hx, flagged_circuit_strs : list[str]):
        
        n_z_stabs = hz.shape[0]
        n_x_stabs = hx.shape[0]
        n_data_qubits = hz.shape[1]
        
        fes = {i : {} for i in range(n_z_stabs + n_x_stabs)}
        
        circs = []
        
        for i_s in range(n_z_stabs + n_x_stabs):
                circ_str = flagged_circuit_strs[i_s]
                circ_str = circ_str.replace('p',f'{0.01}')
                

                circ = stim.Circuit(circ_str)
                
                nc = noisy_circuit(circuit=circ, p=0.01, expanded=True)
                
                expanded_circuit_strs = np.array(nc.__str__().splitlines())
                tags = [literal_eval(l.tag) if l.tag != '' else dict() for l in nc]
                tags_as_dict_of_arrays = list_of_dicts_to_dict_of_arrays(tags)
                circuit_idx_of_efs = np.argwhere(tags_as_dict_of_arrays['typ']=='ef').flatten()
                
                base_mask = np.zeros(len(expanded_circuit_strs),dtype=bool)
                base_mask[np.argwhere(tags_as_dict_of_arrays['typ']!='ef')] = True
                
                for ef in circuit_idx_of_efs:
                        mask = base_mask.copy()
                        mask[ef] = True
                        ef_circuit = stim.Circuit('\n'.join(expanded_circuit_strs[mask].tolist()))
                        
                        simulator = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
                        simulator.do(ef_circuit)
                        mf = simulator.get_measurement_flips(instance_index=0)
                        
                        if mf[-1]:
                                pf = simulator.peek_pauli_flips(instance_index=0)
                                pfd = choose_indices_of_pauli_string(pf, list(range(n_data_qubits)))
                               
                                s = syndrome_of_pauli_string(hx, hz, pfd)
                                
                                if s not in fes[i_s]:
                                        fes[i_s][s] = tuple(pfd)
                                else:
                                        produ = pfd * stim.PauliString(fes[i_s][s])
                                        if produ.weight > 0:
                                                print('same syndrome:')
                                                print(produ)
                            
        return fes

@profile
def generate_flagged_circuits(hz, hx, with_noise : bool = True):
        p = 0.999
        n_z_stabs = hz.shape[0]
        n_x_stabs = hx.shape[0]
        
        first_free_index = hx.shape[1]
        z_stab_indices = [first_free_index + i for i in range(n_z_stabs)]
        first_free_index = max(z_stab_indices)+1
        z_flag_indices = [first_free_index + i for i in range(n_z_stabs)]
        first_free_index = max(z_flag_indices)+1
        x_stab_indices = [first_free_index + i for i in range(n_x_stabs)]
        first_free_index = max(x_stab_indices)+1
        x_flag_indices = [first_free_index + i for i in range(n_x_stabs)]
        first_free_index = max(x_flag_indices)+1
        
        circs = []
        for i_zs in range(n_z_stabs):
                circ = stim.Circuit()
                circ.append('R', z_stab_indices[i_zs])
                circ.append('RX', z_flag_indices[i_zs])
                if with_noise: circ.append('X_ERROR',z_stab_indices[i_zs],p)
                if with_noise: circ.append('Z_ERROR',z_flag_indices[i_zs],p)
                circ.append('TICK')
                s_idx = np.argwhere(hz[i_zs] == 1).flatten()
                for i_t,t in enumerate(s_idx):
                        circ.append('CX', [t,z_stab_indices[i_zs]])
                        if with_noise: circ.append('DEPOLARIZE2', [t,z_stab_indices[i_zs]],p)
                        circ.append('TICK')
                        if i_t == 0 or i_t == len(s_idx)-2:
                                circ.append('CX', [z_flag_indices[i_zs],z_stab_indices[i_zs]])
                                if with_noise: circ.append('DEPOLARIZE2', [z_flag_indices[i_zs],z_stab_indices[i_zs]],p)
                                circ.append('TICK')
                if with_noise: circ.append('X_ERROR',z_stab_indices[i_zs],p)
                if with_noise: circ.append('Z_ERROR',z_flag_indices[i_zs],p)
                circ.append('M', z_stab_indices[i_zs])
                circ.append('MX', z_flag_indices[i_zs])  
                
                circs += [circ.__str__().replace('0.999','p')]    
        
        for i_xs in range(n_x_stabs):
                circ = stim.Circuit()
                circ.append('RX', x_stab_indices[i_xs])
                circ.append('R', x_flag_indices[i_xs])
                if with_noise: circ.append('Z_ERROR',x_stab_indices[i_xs],p)
                if with_noise: circ.append('X_ERROR',x_flag_indices[i_xs],p)
                circ.append('TICK')
                s_idx = np.argwhere(hx[i_xs] == 1).flatten()
                for i_t,t in enumerate(s_idx):
                        circ.append('CX', [x_stab_indices[i_xs],t])
                        if with_noise: circ.append('DEPOLARIZE2', [x_stab_indices[i_xs],t],p)
                        circ.append('TICK')
                        if i_t == 0 or i_t == len(s_idx)-2:
                                circ.append('CX', [x_stab_indices[i_xs],x_flag_indices[i_xs]])
                                if with_noise: circ.append('DEPOLARIZE2', [x_stab_indices[i_xs],x_flag_indices[i_xs]],p)
                                circ.append('TICK')
                if with_noise: circ.append('Z_ERROR',x_stab_indices[i_xs],p)
                if with_noise: circ.append('X_ERROR',x_flag_indices[i_xs],p)
                circ.append('MX', x_stab_indices[i_xs])
                circ.append('M', x_flag_indices[i_xs]) 
                
                circs += [circ.__str__().replace('0.999','p')]     
                
        return circs


class EasiestFlagGate:
        @profile
        def __init__(self, exp: Experiment, gate : str = 's', logical : int = 0, decoder : str = 'lut', noisy_input_layer : bool = True):
                if decoder == 'lookuptable':
                        decoder = 'lut'
                self.decoder = decoder
                self. exp = exp
                self.noisy_input_layer = noisy_input_layer
                # first define the circuits
                self.gate = gate
                self.logical = logical
                if gate in ['s','h']:
                        self.s_gate_flagged = exp.single_qubit_clifford(gate=gate,logical_qubit=logical, noisy=True, flagged=True, stabilizer_measurements=True)
                elif gate == 'cx':
                        self.s_gate_flagged = exp.two_qubit_clifford(gate=gate, logical_qubits=logical, noisy=True, flagged=True, stabilizer_measurements=True)
               
                
                self.s_gate_flagged_noisy = noisy_circuit(self.s_gate_flagged, p=0.01, depolarize_spam=True)
                
                self.sm_flagged = exp.stab_meas(n_rounds=1, noisy=True, flagged=True)
                self.sm_unflagged = exp.stab_meas(n_rounds=1, noisy=True, flagged=False)
                
                self.input_circuit = exp.input_circuit
                self.input_circuit_noisy = noisy_circuit(exp.input_circuit, p=0.01, depolarize_spam=True)
                
                self.sm_flagged_noisy = noisy_circuit(self.sm_flagged, p=0.01, depolarize_spam=True)
                self.sm_unflagged_noisy = noisy_circuit(self.sm_unflagged, p=0.01, depolarize_spam=True)
                
                self.circuit_branch_0_a = exp.encoding_circuit + self.input_circuit + self.s_gate_flagged_noisy  + self.sm_unflagged
                self.circuit_branch_0_b = exp.encoding_circuit + self.input_circuit + self.s_gate_flagged_noisy  + self.sm_unflagged_noisy
                
                self.circuit_branch_1_a = exp.encoding_circuit + self.input_circuit + self.s_gate_flagged + self.sm_flagged_noisy + self.sm_unflagged
                self.circuit_branch_1_b = exp.encoding_circuit + self.input_circuit + self.s_gate_flagged_noisy + self.sm_flagged_noisy + self.sm_unflagged_noisy
                
                # if decoder == 'lut+bposd':
                self.circ_analyzer_0_a = CircuitAnalyzer(self.circuit_branch_0_a)
                self.circ_analyzer_0_b = CircuitAnalyzer(self.circuit_branch_0_b)
                self.circ_analyzer_1_a = CircuitAnalyzer(self.circuit_branch_1_a)
                self.circ_analyzer_1_b = CircuitAnalyzer(self.circuit_branch_1_b)
                
                mo_a = 1
                mo_b = 2
                self.lut_0_a = self.circ_analyzer_0_a.construct_lut(max_order=mo_a, load_if = f'./lut/1_3_{gate}_e_{logical}_0_a_{mo_a}.lut', save_under = f'./lut/1_3_{gate}_e_{logical}_0_a_{mo_a}.lut')
                self.lut_0_b = self.circ_analyzer_0_b.construct_lut(max_order=mo_b, load_if = f'./lut/1_3_{gate}_e_{logical}_0_b_{mo_b}.lut', save_under = f'./lut/1_3_{gate}_e_{logical}_0_b_{mo_b}.lut')
                
                self.lut_1_a = self.circ_analyzer_1_a.construct_lut(max_order=mo_a, load_if = f'./lut/1_3_{gate}_e_{logical}_1_a_{mo_a}.lut', save_under = f'./lut/1_3_{gate}_e_{logical}_1_a_{mo_a}.lut')
                self.lut_1_b = self.circ_analyzer_1_b.construct_lut(max_order=mo_b, load_if = f'./lut/1_3_{gate}_e_{logical}_1_b_{mo_b}.lut', save_under = f'./lut/1_3_{gate}_e_{logical}_1_b_{mo_b}.lut')
                
                
                self.lut_cc = load_lookup_table('./lut/1_3_cc.lut')
        
        @profile
        def run(self, p : float = 0.01, nshots = 1, simulator : stim.FlipSimulator = None, full_circuit : stim.Circuit = None):
                # first construct circuits with correct noise:
                if self.noisy_input_layer:
                        self.input_circuit_noisy = noisy_circuit(self.exp.input_circuit, p=p, depolarize_spam=True)
                
                self.sm_flagged_noisy = noisy_circuit(self.sm_flagged, p=p)
                self.sm_unflagged_noisy = noisy_circuit(self.sm_unflagged, p=p)
                self.s_gate_flagged_noisy = noisy_circuit(self.s_gate_flagged, p=p)
                

                
                # protocol
                final_paulis_on_data = []
                circuit_branches = []
                all_circuits = []
                all_info_strs = []
                iterator = tqdm(range(nshots)) if nshots > 1 else range(nshots)
                for shot in iterator:
                        info_strs = []
                        
                        # if simulator is None:
                        simulator = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
                
                        full_circuit = stim.Circuit()
                        
                        simulator.do(self.exp.encoding_circuit)
                        # full_circuit += self.exp.encoding_circuit
                        if self.noisy_input_layer:
                                simulator.do(self.input_circuit_noisy)
                        # full_circuit += self.input_circuit_noisy
                        
                        simulator.do(self.s_gate_flagged_noisy)
                        # full_circuit += self.s_gate_flagged_noisy
                        dfs = simulator.get_detector_flips(instance_index=0)
                        pauli = simulator.peek_pauli_flips(instance_index=0)
                        # info_strs += ['gate flags : ',f'{ppv(dfs)}',f'{pauli} {pauli.pauli_indices()}']
                        
                        
                        if np.any(dfs):
                                # info_strs += ['* gate flagged - do unflagged sm ec']
                                # for q in range(15,15+12):
                                        # simulator.set_pauli_flip('I', qubit_index=q, instance_index=0)
                                simulator.do(self.sm_unflagged_noisy)
                                # full_circuit += self.sm_unflagged_noisy
                                
                                dfs = simulator.get_detector_flips(instance_index=0)
                                s_1 = dfs[-12:]
                                pauli = simulator.peek_pauli_flips(instance_index=0)
                                
                                # info_strs += [f's_1 = {ppv(s_1)}',f'{pauli} {pauli.pauli_indices()}']

                                full_s = tuple(dfs.astype(int))
                                # info_strs += [f'full_s = {full_s} [{len(full_s)}]']
                                if self.decoder in ['lut+bposd','lut']:
                                        if full_s in self.lut_0_a:
                                                corr = self.lut_0_a[full_s]
                                                # info_strs += [f'corr (of circuit lut):',f'{corr} {corr.pauli_indices()}']
                                        elif full_s in self.lut_0_b:
                                                corr = self.lut_0_b[full_s]
                                        else:
                                                if self.decoder == 'lut':
                                                        corr = stim.PauliString(self.lut_cc.get(tuple(s_1.astype(int)),'I'*15))
                                                elif self.decoder == 'lut+bposd':
                                                        self.bposdDecoder_0 = BpOsdDecoder(self.circ_analyzer_0_b.H, error_rate=float(p))
                                                        corr_int = self.bposdDecoder_0.decode(dfs)
                                                        
                                                        pfs = [self.circ_analyzer_0_b.ef_pauli_flips_data[i] for i in range(len(corr_int)) if corr_int[i] ]
                                                        corr =  reduce(lambda i,j : i*j, pfs)
                                elif self.decoder == 'bposd':
                                        self.bposdDecoder_0 = BpOsdDecoder(self.circ_analyzer_0_b.H, error_rate=float(p))
                                        corr_int = self.bposdDecoder_0.decode(dfs)
                                                        
                                        pfs = [self.circ_analyzer_0_b.ef_pauli_flips_data[i] for i in range(len(corr_int)) if corr_int[i] ]
                                        corr =  reduce(lambda i,j : i*j, pfs)

                                #         corr =  reduce(lambda i,j : i*j, pfs)
                                        # info_strs += [f'corr (of cc lut):',f'{corr} {corr.pauli_indices()}']
                                final_pauli_on_data = choose_indices_of_pauli_string(pauli, list(range(15))) * choose_indices_of_pauli_string(corr, list(range(15))) 
                                # info_strs += [f'final pauli on data:',f'{final_pauli_on_data} {final_pauli_on_data.pauli_indices()}']
                                
                                circuit_branch = 0
                        else:
                                # info_strs += ['* gate did not flag - do ft ec']
                        
                                simulator.do(self.sm_flagged_noisy)
                                # full_circuit += self.sm_flagged_noisy
                                dfs = simulator.get_detector_flips(instance_index=0)
                                
                                zsf = dfs[:12]
                                zs = zsf[:6]
                                zf = zsf[6:]
                                xsf = dfs[12:]
                                xs = xsf[:6]
                                xf = xsf[6:]
                                
                                s_0 = np.concatenate((zs,xs))
                                f_0 = np.concatenate((zf,xf))

                                pauli = simulator.peek_pauli_flips(instance_index=0)
                                
                                # info_strs += [f's_0 = {ppv(s_0)}, f_0 = {ppv(f_0)}',f'{pauli} {pauli.pauli_indices()}']
                                
                        
                                if np.any(dfs):
                                        # info_strs += ['* stab meas 0 flagged - do unflagged circ']
                                        # first set all pauli flips to 0
                                        # for q in range(15,15+12):
                                                # simulator.set_pauli_flip('I', qubit_index=q, instance_index=0)
                                        # simulator.do(stim.Circuit('R ' + ' '.join(f'{q}' for q in range(15,15+12))))
                                        # simulator.do(stim.Circuit('RX ' + ' '.join(f'{q}' for q in range(15,15+12))))
                                        
                                        simulator.do(self.sm_unflagged_noisy)
                                        # full_circuit += self.sm_unflagged_noisy
                                        
                                        dfs = simulator.get_detector_flips(instance_index=0)
                                        s_1 = dfs[-12:]
                                        pauli = simulator.peek_pauli_flips(instance_index=0)
                                        
                                        # info_strs += [f's_1 = {ppv(s_1)}',f'{pauli} {pauli.pauli_indices()}']

                                        full_s = tuple(dfs.astype(int))
                                        # info_strs += [f'full_s = {full_s} [{len(full_s)}]']
                                        if full_s in self.lut_1_a:
                                                corr = self.lut_1_a[full_s]
                                                # info_strs += [f'corr (of circuit lut):',f'{corr} {corr.pauli_indices()}']
                                        elif full_s in self.lut_1_b:
                                                corr = self.lut_1_b[full_s]
                                        else:
                                                if self.decoder == 'lut':
                                                        corr = stim.PauliString(self.lut_cc.get(tuple(s_1.astype(int)),'I'*15))
                                                elif self.decoder == 'lut+bposd':
                                                        self.bposdDecoder_1 = BpOsdDecoder(self.circ_analyzer_1_b.H, error_rate=float(p))
                                                        corr_int = self.bposdDecoder_1.decode(dfs)
                                                        
                                                        pfs = [self.circ_analyzer_1_b.ef_pauli_flips_data[i] for i in range(len(corr_int)) if corr_int[i] ]
                                                        corr =  reduce(lambda i,j : i*j, pfs)
                                                # info_strs += [f'corr (of cc lut):',f'{corr} {corr.pauli_indices()}']
                                        final_pauli_on_data = choose_indices_of_pauli_string(pauli, list(range(15))) * choose_indices_of_pauli_string(corr, list(range(15))) 
                                        # info_strs += [f'final pauli on data:',f'{final_pauli_on_data} {final_pauli_on_data.pauli_indices()}']
                                        
                                        circuit_branch = 1
                                else:
                                        
                                        # do nothing
                                
                                        circuit_branch = 2
                                        
                                        final_pauli_on_data = choose_indices_of_pauli_string(pauli, list(range(15)))
                                        # info_strs += [f'final pauli on data:',f'{final_pauli_on_data} {final_pauli_on_data.pauli_indices()}']
                                        
                                        
                        
                        # all_circuits += [full_circuit]
                        circuit_branches += [circuit_branch]
                        all_info_strs += ['\n'.join(info_strs)]
                        final_paulis_on_data += [final_pauli_on_data]
                
                return final_paulis_on_data, circuit_branches, all_info_strs



class EasyFlagStyleSyndrome:
        def __init__(self, exp : Experiment):
                
                self.exp = exp
                
                # first define the circuits
                self.sm_flagged = exp.stab_meas(n_rounds=1, noisy=True, flagged=True)
                self.sm_unflagged = exp.stab_meas(n_rounds=1, noisy=True, flagged=False)
                
                self.input_circuit_noisy = noisy_circuit(exp.input_circuit, p=0.01)
                
                self.sm_flagged_noisy = noisy_circuit(self.sm_flagged, p=0.01)
                self.sm_unflagged_noisy = noisy_circuit(self.sm_unflagged, p=0.01)
                
                circuit_branch_0 = exp.encoding_circuit + self.input_circuit_noisy + self.sm_flagged_noisy + self.sm_unflagged
                circuit_branch_1 = exp.encoding_circuit + self.input_circuit_noisy + self.sm_flagged + self.sm_flagged_noisy + self.sm_unflagged
                circuit_branch_2 = exp.encoding_circuit + self.sm_flagged + self.input_circuit_noisy + self.sm_flagged  + self.sm_unflagged
                circuit_branch_3 = exp.encoding_circuit + self.input_circuit_noisy + self.sm_flagged + self.sm_flagged
                
                circ_analyzer_0 = CircuitAnalyzer(circuit_branch_0)
                circ_analyzer_1 = CircuitAnalyzer(circuit_branch_1)
                circ_analyzer_2 = CircuitAnalyzer(circuit_branch_2)
                circ_analyzer_3 = CircuitAnalyzer(circuit_branch_3)
                
                self.lut_0 = circ_analyzer_0.construct_lut(max_order=1)
                self.lut_1 = circ_analyzer_1.construct_lut(max_order=1)
                self.lut_2 = circ_analyzer_2.construct_lut(max_order=1)
                self.lut_3 = circ_analyzer_3.construct_lut(max_order=1)
                
                self.lut_cc = load_lookup_table('./lut/1_3_cc.lut')
        
                
                
                self.circuits = [circuit_branch_0, circuit_branch_1, circuit_branch_2, circuit_branch_3]
                self.luts = [self.lut_0,self.lut_1,self.lut_2, self.lut_3]
        
        def run(self, p : float = 0.01, nshots = 1, simulator : stim.FlipSimulator = None, full_circuit : stim.Circuit = None):
                # first construct circuits with correct noise:
                self.input_circuit_noisy = noisy_circuit(self.exp.input_circuit, p=p)
                
                self.sm_flagged_noisy = noisy_circuit(self.sm_flagged, p=p)
                self.sm_unflagged_noisy = noisy_circuit(self.sm_unflagged, p=p)
                
                # protocol
                final_paulis_on_data = []
                circuit_branches = []
                all_circuits = []
                all_info_strs = []
                iterator = tqdm(range(nshots)) if nshots > 1 else range(nshots)
                for shot in iterator:
                        info_strs = []
                        
                        # if simulator is None:
                        simulator = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
                
                        full_circuit = stim.Circuit()
                        
                        simulator.do(self.exp.encoding_circuit)
                        # full_circuit += self.exp.encoding_circuit
                        
                        simulator.do(self.input_circuit_noisy)
                        # full_circuit += self.input_circuit_noisy
                        
                        simulator.do(self.sm_flagged_noisy)
                        # full_circuit += self.sm_flagged_noisy
                        dfs = simulator.get_detector_flips(instance_index=0)
                        
                        zsf = dfs[:12]
                        zs = zsf[:6]
                        zf = zsf[6:]
                        xsf = dfs[12:]
                        xs = xsf[:6]
                        xf = xsf[6:]
                        
                        s_0 = np.concatenate((zs,xs))
                        f_0 = np.concatenate((zf,xf))

                        pauli = simulator.peek_pauli_flips(instance_index=0)
                        
                        # info_strs += [f's_0 = {ppv(s_0)}, f_0 = {ppv(f_0)}',f'{pauli} {pauli.pauli_indices()}']
                        
                       
                        if np.any(f_0):
                                # info_strs += ['* stab meas 0 flagged - do unflagged circ']
                                # first set all pauli flips to 0
                                # for q in range(15,15+12):
                                        # simulator.set_pauli_flip('I', qubit_index=q, instance_index=0)
                                # simulator.do(stim.Circuit('R ' + ' '.join(f'{q}' for q in range(15,15+12))))
                                # simulator.do(stim.Circuit('RX ' + ' '.join(f'{q}' for q in range(15,15+12))))
                                
                                simulator.do(self.sm_unflagged_noisy)
                                # full_circuit += self.sm_unflagged_noisy
                                
                                dfs = simulator.get_detector_flips(instance_index=0)
                                s_1 = dfs[-12:]
                                pauli = simulator.peek_pauli_flips(instance_index=0)
                                
                                # info_strs += [f's_1 = {ppv(s_1)}',f'{pauli} {pauli.pauli_indices()}']

                                full_s = tuple(dfs.astype(int))
                                # info_strs += [f'full_s = {full_s} [{len(full_s)}]']
                                if full_s in self.lut_0:
                                        corr = self.lut_0[full_s]
                                        # info_strs += [f'corr (of circuit lut):',f'{corr} {corr.pauli_indices()}']
                                else:
                                        corr = stim.PauliString(self.lut_cc.get(tuple(s_1.astype(int)), 'I'*15))
                                        # info_strs += [f'corr (of cc lut):',f'{corr} {corr.pauli_indices()}']
                                final_pauli_on_data = choose_indices_of_pauli_string(pauli, list(range(15))) * choose_indices_of_pauli_string(corr, list(range(15))) 
                                # info_strs += [f'final pauli on data:',f'{final_pauli_on_data} {final_pauli_on_data.pauli_indices()}']
                                
                                circuit_branch = 0
                        else:
                                # info_strs += ['* stab meas 0 did not flag - do flagged circ again']
                                # first set all pauli flips to 0
                                # for q in range(15,15+12):
                                        # simulator.set_pauli_flip('I', qubit_index=q, instance_index=0)
                                # simulator.do(stim.Circuit('R ' + ' '.join(f'{q}' for q in range(15,15+12))))
                                # simulator.do(stim.Circuit('RX ' + ' '.join(f'{q}' for q in range(15,15+12))))
                                
                                simulator.do(self.sm_flagged_noisy)
                                # full_circuit += self.sm_flagged_noisy
                                dfs = simulator.get_detector_flips(instance_index=0)
                                
                                zsf = dfs[-24:-12]
                                zs = zsf[:6]
                                zf = zsf[6:]
                                xsf = dfs[-12:]
                                xs = xsf[:6]
                                xf = xsf[6:]
                                
                                s_1 = np.concatenate((zs,xs))
                                f_1 = np.concatenate((zf,xf))
                                

                                pauli = simulator.peek_pauli_flips(instance_index=0)
                                
                                # info_strs += [f's_1 = {ppv(s_1)}, f_1 = {ppv(f_1)}',f'{pauli} {pauli.pauli_indices()}']
                                
                                if np.any(f_1):
                                        # info_strs += ['* stab meas 1 flagged - do unflagged circ']
                                        # first set all pauli flips to 0
                                        # for q in range(15,15+12):
                                                # simulator.set_pauli_flip('I', qubit_index=q, instance_index=0)
                                        # simulator.do(stim.Circuit('R ' + ' '.join(f'{q}' for q in range(15,15+12))))
                                        # simulator.do(stim.Circuit('RX ' + ' '.join(f'{q}' for q in range(15,15+12))))
                                        
                                        dfs_bf  = tuple(simulator.get_detector_flips(instance_index=0).astype(int))
                                        # info_strs += [f'dfs_bf = {dfs_bf} [{len(dfs_bf)}]']
                                        
                                        simulator.do(self.sm_unflagged_noisy)
                                        # full_circuit += self.sm_unflagged_noisy
                                        
                                        dfs = simulator.get_detector_flips(instance_index=0)
                                        s_2 = dfs[-12:]
                                        pauli = simulator.peek_pauli_flips(instance_index=0)
                                        
                                        # info_strs += [f's_2 = {ppv(s_2)}',f'{pauli} {pauli.pauli_indices()}']
                                        
                                        full_s = tuple(dfs.astype(int))
                                        # info_strs += [f'full_s = {full_s} [{len(full_s)}]']
                                        if full_s in self.lut_1:
                                                corr = self.lut_1[full_s]
                                                # info_strs += [f'corr (of circuit lut):',f'{corr} {corr.pauli_indices()}']
                                        else:
                                                corr = stim.PauliString(self.lut_cc.get(tuple(s_2.astype(int)), 'I'*15))
                                                # info_strs += [f'corr (of cc lut):',f'{corr} {corr.pauli_indices()}']
                                        
                                        final_pauli_on_data = choose_indices_of_pauli_string(pauli, list(range(15))) * choose_indices_of_pauli_string(corr, list(range(15))) 
                                        # info_strs += [f'final pauli on data:',f'{final_pauli_on_data} {final_pauli_on_data.pauli_indices()}']
                                        
                                        circuit_branch = 1
                                else:
                                        # info_strs += ['* stab meas 1 did not flag - ']                                        
                                        
                                        if np.any(s_0 != s_1):
                                                # info_strs += [' - syndromes disagree: do round of unflagged']
                                                
                                                simulator.do(self.sm_unflagged_noisy)
                                                # full_circuit += self.sm_unflagged_noisy
                                                
                                                dfs = simulator.get_detector_flips(instance_index=0)
                                                s_2 = dfs[-12:]
                                                pauli = simulator.peek_pauli_flips(instance_index=0)
                                                
                                                # info_strs += [f's_2 = {ppv(s_2)}',f'{pauli} {pauli.pauli_indices()}']

                                                full_s = tuple(dfs.astype(int))
                                                # info_strs += [f'full_s = {full_s} [{len(full_s)}]']
                                                if full_s in self.lut_2:
                                                        corr = self.lut_2[full_s]
                                                        # info_strs += [f'corr (of circuit lut):',f'{corr} {corr.pauli_indices()}']
                                                else:
                                                        corr = stim.PauliString(self.lut_cc.get(tuple(s_2.astype(int)), 'I'*15))
                                                        # info_strs += [f'corr (of cc lut):',f'{corr} {corr.pauli_indices()}']
                                                circuit_branch = 2
                                        
                                        else:   
                                                # info_strs += [' - syndromes agree: do nothing an directly correct']
                                        
                                                full_s = tuple(dfs.astype(int))
                                                # info_strs += [f'full_s = {full_s} [{len(full_s)}]']
                                                if full_s in self.lut_3:
                                                        corr = self.lut_3[full_s]
                                                        # info_strs += [f'corr (of circuit lut):',f'{corr} {corr.pauli_indices()}']
                                                else:
                                                        corr = stim.PauliString(self.lut_cc.get(tuple(s_1.astype(int)), 'I'*15))
                                                        # info_strs += [f'corr (of cc lut):',f'{corr} {corr.pauli_indices()}']
                                                
                                                circuit_branch = 3
                                        
                                        final_pauli_on_data = choose_indices_of_pauli_string(pauli, list(range(15))) * choose_indices_of_pauli_string(corr, list(range(15))) 
                                        # info_strs += [f'final pauli on data:',f'{final_pauli_on_data} {final_pauli_on_data.pauli_indices()}']
                                        
                                        
                        
                        # all_circuits += [full_circuit]
                        circuit_branches += [circuit_branch]
                        all_info_strs += ['\n'.join(info_strs)]
                        final_paulis_on_data += [final_pauli_on_data]
                
                return final_paulis_on_data, circuit_branches, all_info_strs#, all_circuits

class EasiestFlagStyleSyndrome:
        def __init__(self, exp : Experiment):
                
                self.exp = exp
                
                # first define the circuits
                self.sm_flagged = exp.stab_meas(n_rounds=1, noisy=True, flagged=True)
                self.sm_unflagged = exp.stab_meas(n_rounds=1, noisy=True, flagged=False)
                
                self.input_circuit = exp.input_circuit
                self.input_circuit_noisy = noisy_circuit(exp.input_circuit, p=0.01)
                
                self.sm_flagged_noisy = noisy_circuit(self.sm_flagged, p=0.01)
                self.sm_unflagged_noisy = noisy_circuit(self.sm_unflagged, p=0.01)
                
                circuit_branch_0_a = exp.encoding_circuit + self.input_circuit_noisy + self.sm_flagged_noisy + self.sm_unflagged
                circuit_branch_0_b = exp.encoding_circuit + self.input_circuit_noisy + self.sm_flagged_noisy + self.sm_unflagged_noisy
                circuit_branch_0_c = exp.encoding_circuit + self.sm_unflagged_noisy + self.sm_unflagged
                
                circ_analyzer_0_a = CircuitAnalyzer(circuit_branch_0_a)
                circ_analyzer_0_b = CircuitAnalyzer(circuit_branch_0_b)
                circ_analyzer_0_c = CircuitAnalyzer(circuit_branch_0_c)
                
                 # if decoder == 'lut+bposd':
                
                self.lut_0_a = circ_analyzer_0_a.construct_lut(max_order=2, load_if=f'./lut/1_3_ec_0_a.lut', save_under=f'./lut/1_3_ec_0_a.lut')
                self.lut_0_b = circ_analyzer_0_b.construct_lut(max_order=2, load_if=f'./lut/1_3_ec_0_b.lut', save_under=f'./lut/1_3_ec_0_b.lut')
                
                # self.lut_cc = circ_analyzer_0_c.construct_lut(max_order=15, restrict_decs = -12)
                self.lut_cc = load_lookup_table('./lut/1_3_cc.lut')        
                
                self.circuits = [circuit_branch_0_a, circuit_branch_0_b]
                self.luts = [self.lut_0_a, self.lut_0_b]
        
        def run(self, p : float = 0.01, nshots = 1, simulator : stim.FlipSimulator = None, full_circuit : stim.Circuit = None):
                # first construct circuits with correct noise:
                self.input_circuit_noisy = noisy_circuit(self.input_circuit, p=p)
                
                self.sm_flagged_noisy = noisy_circuit(self.sm_flagged, p=p)
                self.sm_unflagged_noisy = noisy_circuit(self.sm_unflagged, p=p)
                
                # protocol
                final_paulis_on_data = []
                circuit_branches = []
                all_circuits = []
                all_info_strs = []
                iterator = tqdm(range(nshots)) if nshots > 1 else range(nshots)
                for shot in iterator:
                        info_strs = []
                        
                        # if simulator is None:
                        simulator = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
                
                        # full_circuit = stim.Circuit()
                        
                        simulator.do(self.exp.encoding_circuit)
                        # full_circuit += self.exp.encoding_circuit
                        
                        simulator.do(self.input_circuit_noisy)
                        # full_circuit += self.input_circuit_noisy
                        
                        simulator.do(self.sm_flagged_noisy)
                        # full_circuit += self.sm_flagged_noisy
                        dfs = simulator.get_detector_flips(instance_index=0)
                        
                        zsf = dfs[:12]
                        zs = zsf[:6]
                        zf = zsf[6:]
                        xsf = dfs[12:]
                        xs = xsf[:6]
                        xf = xsf[6:]
                        
                        s_0 = np.concatenate((zs,xs))
                        f_0 = np.concatenate((zf,xf))

                        pauli = simulator.peek_pauli_flips(instance_index=0)
                        
                        info_strs += [f's_0 = {ppv(s_0)}, f_0 = {ppv(f_0)}',f'{pauli} {pauli.pauli_indices()}']
                        
                       
                        if np.any(dfs):
                                # info_strs += ['* stab meas 0 flagged - do unflagged circ']
                                # first set all pauli flips to 0
                                # for q in range(15,15+12):
                                        # simulator.set_pauli_flip('I', qubit_index=q, instance_index=0)
                                # simulator.do(stim.Circuit('R ' + ' '.join(f'{q}' for q in range(15,15+12))))
                                # simulator.do(stim.Circuit('RX ' + ' '.join(f'{q}' for q in range(15,15+12))))
                                
                                simulator.do(self.sm_unflagged_noisy)
                                # full_circuit += self.sm_unflagged_noisy
                                
                                dfs = simulator.get_detector_flips(instance_index=0)
                                s_1 = dfs[-12:]
                                pauli = simulator.peek_pauli_flips(instance_index=0)
                                
                                info_strs += [f's_1 = {ppv(s_1)}',f'{pauli} {pauli.pauli_indices()}']

                                full_s = tuple(dfs.astype(int))
                                info_strs += [f'full_s = {full_s} [{len(full_s)}]']
                                if full_s in self.lut_0_a:
                                        corr = self.lut_0_a[full_s]
                                        info_strs += [f'corr (of circuit lut a):',f'{corr} {corr.pauli_indices()}']
                                elif full_s in self.lut_0_b:
                                        corr = self.lut_0_b[full_s]
                                        info_strs += [f'corr (of circuit lut b):',f'{corr} {corr.pauli_indices()}']
                                else:
                                        corr = stim.PauliString(self.lut_cc.get(tuple(s_1.astype(int)), 'I'*15))
                                        info_strs += [f'corr (of cc lut):',f'{corr} {corr.pauli_indices()}']
                                final_pauli_on_data = choose_indices_of_pauli_string(pauli, list(range(15))) * choose_indices_of_pauli_string(corr, list(range(15))) 
                                info_strs += [f'final pauli on data:',f'{final_pauli_on_data} {final_pauli_on_data.pauli_indices()}']
                                
                                circuit_branch = 0
                        else:
                                # do nothing
                                circuit_branch = 1
                                
                                
                                final_pauli_on_data = choose_indices_of_pauli_string(pauli, list(range(15)))
                                info_strs += [f'final pauli on data:',f'{final_pauli_on_data} {final_pauli_on_data.pauli_indices()}']
                                        
                                        
                        
                        # all_circuits += [full_circuit]
                        circuit_branches += [circuit_branch]
                        all_info_strs += ['\n'.join(info_strs)]
                        final_paulis_on_data += [final_pauli_on_data]
                
                return final_paulis_on_data, circuit_branches, all_info_strs#, all_circuits

@profile
def flag_style_syndrome(flagged_circuits : list[stim.Circuit], perfect_encoding_circuit : stim.Circuit, efficient_ec_circuit : stim.Circuit, hx, hz, p, t : int = 1, flagged : bool = False, max_repetitions : int = None, flag_error_set : dict = {}, lut : dict = None, simulator : stim.FlipSimulator = None):
        if simulator is None:
                simulator = stim.FlipSimulator(batch_size=1,disable_stabilizer_randomization=True)
                simulator.do(perfect_encoding_circuit)
        
        if max_repetitions is None:
                max_repetitions = (t+1)**2
        same_measurements = 0
        n_repetitions = 0
        n_z_stabs = hz.shape[0]
        n_x_stabs = hx.shape[0]
        
        first_free_index = perfect_encoding_circuit.num_qubits
        z_stab_indices = [first_free_index + i for i in range(n_z_stabs)]
        first_free_index = max(z_stab_indices)+1
        z_flag_indices = [first_free_index + i for i in range(n_z_stabs)]
        first_free_index = max(z_flag_indices)+1
        x_stab_indices = [first_free_index + i for i in range(n_x_stabs)]
        first_free_index = max(x_stab_indices)+1
        x_flag_indices = [first_free_index + i for i in range(n_x_stabs)]
        first_free_index = max(x_flag_indices)+1
        
        info_str = ''
        
        flagged = False
        s = {0 : [], 1 : []}
        # measure stabilizer generators, first z then x, flagged:
        for n_repetitions in range(2):
                info_str += f'{n_repetitions}: '
                for i_s in range(n_z_stabs + n_x_stabs):                        
                        circ = flagged_circuits[i_s]
                        
                        simulator.do(circ)
                
                        measurement_flips = simulator.get_measurement_flips(instance_index=0)

                        s[n_repetitions] += [measurement_flips[-2]]
                        
                        if measurement_flips[-1]:
                                pauli_flips = simulator.peek_pauli_flips(instance_index=0)
                                flagged=True
                                info_str += f'{i_s} flagged\n'
                                break
                                
                if flagged:
                        break

        do_perfect_round = False
        if flagged:
                info_str += f'flagged\n'
                info_str += f'pauli b/f perf: {choose_indices_of_pauli_string(pauli_flips, list(range(15)))} \n' 
                do_perfect_round = True
        else:
                info_str += f'two reps without flags, s = '
                info_str += f''.join([f'{int(q)}' for q in s[0]]) + '\n'
                if np.all(s[0] == s[1]):
                        info_str += f'both syndromes same\n'
                        sz = np.array(s[1][:n_z_stabs])
                        sx = np.array(s[1][n_z_stabs:])
                        s = s[1]
                else:
                        info_str += f'syndromes differ\n'
                        do_perfect_round = True

        if do_perfect_round:
                simulator.do(efficient_ec_circuit)
                
                s = simulator.get_measurement_flips(instance_index=0)[-(n_z_stabs+n_x_stabs):]

                sz = s[:n_z_stabs]
                sx = s[-n_x_stabs:]
                info_str += 'non-flagged round: \n'
                info_str += ' sz: ' + ''.join(sz.astype(int).astype(str)) + '\t' + ' sx: ' + ''.join(sx.astype(int).astype(str)) + '\n'

        
        pauli_on_data = choose_indices_of_pauli_string(simulator.peek_pauli_flips(instance_index=0),list(range(15)))
       
        if sum(s) == 0:
                return pauli_on_data, info_str

        s_lut = tuple(int(ss) for ss in s)

        info_str += f's_lut : {s_lut}\n'
        if flagged and s_lut in flag_error_set[i_s]:
                info_str += 's_lut in flag_error_set'
                final_pauli_correction_on_data = stim.PauliString(flag_error_set[i_s][s_lut])
        else:
                if lut is None:
                        decoder_hz = BpOsdDecoder(pcm=hz, error_rate=float(p))
                        decoder_hx = BpOsdDecoder(pcm=hx, error_rate=float(p))

                        correction_x = decoder_hz.decode(sz)
                        correction_z = decoder_hx.decode(sx)
                        
                        corr_p = stim.PauliString.from_numpy(xs=correction_x,zs=correction_z)
                else:
                        corr_p = stim.PauliString(lut[s_lut])
                
                final_pauli_correction_on_data = choose_indices_of_pauli_string(corr_p, list(range(15)))
                
        final_pauli_on_data = pauli_on_data*final_pauli_correction_on_data
        
        info_str += f'pauli_on_data = {pauli_on_data}, {pauli_on_data.pauli_indices()}\n'
        info_str += f'corr_on_data  = {final_pauli_correction_on_data}, {final_pauli_correction_on_data.pauli_indices()}\n'
        info_str += f'rerr_on_data  = {final_pauli_on_data}, {final_pauli_on_data.pauli_indices()}\n'

       
        return final_pauli_on_data, info_str

def gt(channel : int, typ : str = 'ef') -> str:
        return str({'channel' : channel, 'typ' : typ})

def interface_circuit(p : float, stabilizer_data_indices : list[int],interface_data_idx : int | list[int], stabilizer_measurement_idx : int, stabilizer_flag_idx : int, interface_flag_idx : int | list[int], channel_index : int = 0) -> tuple[stim.Circuit,stim.Circuit]:
        circ_a = stim.Circuit()
        circ_b = stim.Circuit()
        
        if not isinstance(interface_data_idx, list):
                interface_data_idx = [interface_data_idx]
        if not isinstance(interface_flag_idx, list):
                interface_flag_idx = [interface_flag_idx]
        assert len(interface_data_idx) == len(interface_flag_idx), 'len(interface_data_idx) != len(interface_flag_idx)'
        
        # inits
        circ_a.append_from_stim_program_text(f'R {stabilizer_measurement_idx} ' + ' '.join([f'{i}' for i in interface_flag_idx]))
        circ_a.append_from_stim_program_text(f'RX {stabilizer_flag_idx}')
        circ_a.append_from_stim_program_text(f'X_ERROR[{gt(channel_index)}]({p}) {stabilizer_measurement_idx} ' + ' '.join([f'{i}' for i in interface_flag_idx]))
        channel_index += 1
        circ_a.append_from_stim_program_text(f'Z_ERROR[{gt(channel_index)}]({p}) {stabilizer_flag_idx}')
        circ_a.append('TICK')
        
        # open interface flag
        for i in range(len(interface_data_idx)):
                circ_a.append_from_stim_program_text(f'CX {interface_data_idx[i]} {interface_flag_idx[i]}')
                circ_a.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {interface_data_idx[i]} {interface_flag_idx[i]}')
                channel_index += 1
        circ_a.append('TICK')
        
        # do flagged stab gates
        for ic,c in enumerate(stabilizer_data_indices):
                circ_a.append_from_stim_program_text(f'CX {c} {stabilizer_measurement_idx}')
                circ_a.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {c} {stabilizer_measurement_idx}') 
                channel_index += 1
                circ_a.append('TICK')
                
                if ic in [0, len(stabilizer_data_indices)-2]:
                        circ_a.append_from_stim_program_text(f'CX {stabilizer_flag_idx} {stabilizer_measurement_idx}')
                        circ_a.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {stabilizer_flag_idx} {stabilizer_measurement_idx}') 
                        channel_index += 1
                        circ_a.append('TICK')
        # stab measurement
        circ_a.append_from_stim_program_text(f'X_ERROR[{gt(channel_index)}]({p}) {stabilizer_measurement_idx}')
        channel_index += 1
        circ_a.append_from_stim_program_text(f'Z_ERROR[{gt(channel_index)}]({p}) {stabilizer_flag_idx}')
        circ_a.append_from_stim_program_text(f'M {stabilizer_measurement_idx}')
        circ_a.append_from_stim_program_text('DETECTOR rec[-1]')
        circ_a.append_from_stim_program_text(f'MX {stabilizer_flag_idx}')
        circ_a.append_from_stim_program_text('DETECTOR rec[-1]')
        
        # close interface flag
        for i in range(len(interface_data_idx)):
                circ_b.append_from_stim_program_text(f'CX {interface_data_idx[i]} {interface_flag_idx[i]}')
                circ_b.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {interface_data_idx[i]} {interface_flag_idx[i]}')
                channel_index += 1
        circ_b.append('TICK')
        # measure interface flags
        for i in range(len(interface_data_idx)):
                circ_b.append_from_stim_program_text(f'X_ERROR[{gt(channel_index)}]({p}) {interface_flag_idx[i]}')
                channel_index += 1
                circ_b.append_from_stim_program_text(f'M {interface_flag_idx[i]}')
                circ_b.append_from_stim_program_text('DETECTOR rec[-1]')
        
        return circ_a, circ_b, channel_index


def coloration_syndrome_cycle(min_edge_coloration : list[list[tuple]], data_qubits : list[int], ancilla_qubits : list[int], pauli : str = 'X', with_init : bool = True, p : float = None , tag : str = '', channel_index : int = 0) -> tuple[stim.Circuit,int]:
        circuit = stim.Circuit()
        op = {'X':'Z','Z':'X'}
        ancilla_pos = 0 
        last_appeared = {}
        for color,cnots in enumerate(min_edge_coloration):
                # print(color)
                for cnot in cnots:
                        ancilla = cnot[ancilla_pos]
                        last_appeared[ancilla]=color#print(ancilla)

        if with_init:
                circuit.append(f'R{pauli}',[q for q in ancilla_qubits], tag=tag)
                if p is not None:
                        circuit.append(f'{op[pauli]}_ERROR',[q for q in ancilla_qubits], p, tag=gt(channel_index))
                        channel_index += 1
        
        circuit.append('TICK')
        
        for color,cnots in enumerate(min_edge_coloration):
                ancillas_in_color = []
                for i_c,cnot in enumerate(cnots):
                        CNOT = [cnot[1],cnot[0]] if pauli == 'Z' else [cnot[0],cnot[1]]
                        circuit.append('CX',CNOT, tag=tag)
                        if p is not None:
                                circuit.append('DEPOLARIZE2',CNOT, p, tag=gt(channel_index))
                                channel_index += 1
                        ancillas_in_color += [cnot[ancilla_pos]]
                circuit.append('TICK')
                
                        
                for ancilla in ancilla_qubits:
                        if last_appeared[ancilla] == color:
                                if p is not None:
                                        circuit.append(f'{op[pauli]}_ERROR',ancilla, p, tag=gt(channel_index))
                                        channel_index += 1
                                circuit.append(f'M{pauli}',ancilla, tag=tag) 
                                circuit.append_from_stim_program_text('DETECTOR rec[-1]')
                

        return circuit, channel_index

def stab_meas_circuit(p : float, hx, hz, first_free_index : int, channel_index : int = 0) -> tuple[stim.Circuit,int,int]:
        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        data_qubits = list(range(hz.shape[1]))
        
        x_ancilla_qubits = [first_free_index + i for i in range(n_x_stabs)]
        first_free_index = max(x_ancilla_qubits) + 1
        z_ancilla_qubits = [first_free_index + i for i in range(n_z_stabs)]
        first_free_index = max(z_ancilla_qubits) + 1
        
        min_edge_coloraton_z = coloration_schedule(hz, check_offset=min(z_ancilla_qubits), coloring='naive')
        min_edge_coloraton_x = coloration_schedule(hx, check_offset=min(x_ancilla_qubits), coloring='naive')
        
        circuit = stim.Circuit()
        # measure Z stabilizers
        z_stabs, channel_index = coloration_syndrome_cycle(min_edge_coloraton_z,data_qubits,z_ancilla_qubits,'Z',p=p, channel_index = channel_index)
        circuit += z_stabs
                
        # measure X stabilizers
        x_stabs, channel_index = coloration_syndrome_cycle(min_edge_coloraton_x,data_qubits,x_ancilla_qubits,'X',p=p, channel_index = channel_index)
        circuit += x_stabs
        
        return circuit, first_free_index, channel_index
        

def unflagged_s_circ(p : float, logical_idx : list[int], channel_index : int = 0) -> tuple[stim.Circuit, int]:
        s_circ = stim.Circuit()
        s_circ.append_from_stim_program_text(f'CZ {logical_idx[0]} {logical_idx[1]}')
        s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[0]} {logical_idx[1]}')
        channel_index += 1
        s_circ.append_from_stim_program_text(f'TICK')
        s_circ.append_from_stim_program_text(f'CZ {logical_idx[0]} {logical_idx[2]}')
        s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[0]} {logical_idx[2]}')
        channel_index += 1
        s_circ.append_from_stim_program_text(f'TICK')
        s_circ.append_from_stim_program_text(f'CZ {logical_idx[1]} {logical_idx[2]}')
        s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[1]} {logical_idx[2]}')
        channel_index += 1
        s_circ.append_from_stim_program_text(f'TICK')
        for i in range(len(logical_idx)):
                s_circ.append_from_stim_program_text(f'S {logical_idx[i]}')
                s_circ.append_from_stim_program_text(f'DEPOLARIZE1[{gt(channel_index)}]({p}) {logical_idx[i]}')
                channel_index += 1
        return s_circ, channel_index


def ppv(vector) -> str:
        return ''.join(vector.astype(int).astype(str))

def flagged_s_gate(code : LCSCode, s_gate : int = 0, unitary_encoding : stim.Circuit = None, p : float = 0.01, nshots : int = 1):
        
        logical_idx = code.logical_indices['Z'][s_gate]
        # info_strss += [f'logical indices: {logical_idx}']
        
        
        # first construct all possible relevant circuits
        
        ## First stabilizer to be measured (overlaps with first blocks of left and right qubits)
        stab_1 = code.stabilizer_indices['Z'][s_gate]
        ifd_1_idx = logical_idx[0]
        sm_1_idx = 15
        sf_1_idx = 16
        if_1_idx = 17
        first_free_idx = 18
        
        channel_index = 0
        
        circ_stab_1_a, circ_stab_1_b, channel_index = interface_circuit(p=p, stabilizer_data_indices=stab_1, interface_data_idx = ifd_1_idx, stabilizer_measurement_idx=sm_1_idx, stabilizer_flag_idx=sf_1_idx, interface_flag_idx=if_1_idx, channel_index=channel_index)
        
        
        # Second stabilizer to be measured (overlaps with 2nd block of left (and also first of right) qubits)
        stab_2 = code.stabilizer_indices['Z'][s_gate+3]
        ifd_2_idx = [logical_idx[1],logical_idx[2]]
        sm_2_idx = first_free_idx
        sf_2_idx = first_free_idx+1
        if_2_idx = [first_free_idx+2,first_free_idx+3]
        first_free_idx = max(if_2_idx) + 1
        
        circ_stab_2_a, circ_stab_2_b, channel_index = interface_circuit(p=p, stabilizer_data_indices=stab_2, interface_data_idx = ifd_2_idx, stabilizer_measurement_idx=sm_2_idx, stabilizer_flag_idx=sf_2_idx, interface_flag_idx=if_2_idx, channel_index=channel_index)
        
        
        ## unprotected s circ
        s_circ, channel_index = unflagged_s_circ(p=p, logical_idx=logical_idx, channel_index=channel_index)
        
        ## first row flagged s circ
        flag_for_x = first_free_idx # the X flag is an interface flag now
        flag_for_z_1 = first_free_idx+1
        flag_for_z_2 = flag_for_z_1+1
        first_free_idx = flag_for_z_2+1
        
        first_row_s_circ = stim.Circuit()
        
        
        # flag for Z around gate 1
        first_row_s_circ.append_from_stim_program_text(f'RX {flag_for_z_1}')
        first_row_s_circ.append_from_stim_program_text(f'Z_ERROR[{gt(channel_index)}]({p}) {flag_for_z_1}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        first_row_s_circ.append_from_stim_program_text(f'CX {flag_for_z_1} {logical_idx[1]}')
        first_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {flag_for_z_1} {logical_idx[1]}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        
        # Gate part 1
        first_row_s_circ.append_from_stim_program_text(f'CZ {logical_idx[0]} {logical_idx[1]}')
        first_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[0]} {logical_idx[1]}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        
        
        # flag for Z around gate 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        first_row_s_circ.append_from_stim_program_text(f'CZ {flag_for_z_1} {logical_idx[0]}')
        first_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {flag_for_z_1} {logical_idx[0]}')
        channel_index += 1
        
        first_row_s_circ.append_from_stim_program_text(f'CX {flag_for_z_1} {logical_idx[1]}')
        first_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {flag_for_z_1} {logical_idx[1]}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        
        first_row_s_circ.append_from_stim_program_text(f'Z_ERROR[{gt(channel_index)}]({p}) {flag_for_z_1}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'MX {flag_for_z_1}')
        first_row_s_circ.append_from_stim_program_text(f'DETECTOR rec[-1]')
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        
        # close X interface flag
        first_row_s_circ += circ_stab_1_b
        
        
        # flag for X around gate 2
        first_row_s_circ.append_from_stim_program_text(f'R {flag_for_x}')
        first_row_s_circ.append_from_stim_program_text(f'X_ERROR[{gt(channel_index)}]({p}) {flag_for_x}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        first_row_s_circ.append_from_stim_program_text(f'CX {logical_idx[2]} {flag_for_x}')
        first_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[2]} {flag_for_x}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        
        
        # flag for Z around gate 2
        first_row_s_circ.append_from_stim_program_text(f'RX {flag_for_z_2}')
        first_row_s_circ.append_from_stim_program_text(f'Z_ERROR[{gt(channel_index)}]({p}) {flag_for_z_2}')
        channel_index += 1
        
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        first_row_s_circ.append_from_stim_program_text(f'CX {flag_for_z_2} {logical_idx[0]}')
        first_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {flag_for_z_2} {logical_idx[0]}')
        channel_index += 1
        
        # Gate 2
        first_row_s_circ.append_from_stim_program_text(f'CZ {logical_idx[0]} {logical_idx[2]}')
        first_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[0]} {logical_idx[2]}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        first_row_s_circ.append_from_stim_program_text(f'CZ {flag_for_z_2} {logical_idx[2]}')
        first_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {flag_for_z_2} {logical_idx[2]}')


        # flag for Z around gate 2
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        first_row_s_circ.append_from_stim_program_text(f'CX {flag_for_z_2} {logical_idx[0]}')
        first_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {flag_for_z_2} {logical_idx[0]}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        first_row_s_circ.append_from_stim_program_text(f'Z_ERROR[{gt(channel_index)}]({p}) {flag_for_z_2}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'MX {flag_for_z_2}')
        first_row_s_circ.append_from_stim_program_text(f'DETECTOR rec[-1]')
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        
        # flag for X around gate 2
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        first_row_s_circ.append_from_stim_program_text(f'CX {logical_idx[2]} {flag_for_x}')
        first_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[2]} {flag_for_x}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'TICK')
        first_row_s_circ.append_from_stim_program_text(f'X_ERROR[{gt(channel_index)}]({p}) {flag_for_x}')
        channel_index += 1
        first_row_s_circ.append_from_stim_program_text(f'M {flag_for_x}')
        first_row_s_circ.append_from_stim_program_text(f'DETECTOR rec[-1]')
        
        ## second row s circ unflagged
        second_row_s_circ = stim.Circuit()
        second_row_s_circ.append_from_stim_program_text(f'CZ {logical_idx[1]} {logical_idx[2]}')
        second_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[1]} {logical_idx[2]}')
        channel_index += 1
        second_row_s_circ.append_from_stim_program_text(f'TICK')
        for i in range(len(logical_idx)):
                second_row_s_circ.append_from_stim_program_text(f'S {logical_idx[i]}')
                second_row_s_circ.append_from_stim_program_text(f'DEPOLARIZE1[{gt(channel_index)}]({p}) {logical_idx[i]}')
                channel_index += 1
        
        
        ## second row s circ flagged
        flag_for_x = if_2_idx # the X flag is an interface flag now
        flag_for_x_2 = first_free_idx # the X flag is an interface flag now
        flag_for_z = first_free_idx +1
        first_free_idx = flag_for_z+1
        
        second_row_s_circ_flagged = stim.Circuit()
        
        # flag for x
        second_row_s_circ_flagged.append_from_stim_program_text(f'RZ {flag_for_x_2}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'X_ERROR[{gt(channel_index)}]({p}) {flag_for_x_2}')
        channel_index += 1
        second_row_s_circ_flagged.append_from_stim_program_text(f'TICK')
        second_row_s_circ_flagged.append_from_stim_program_text(f'CX {logical_idx[2]} {flag_for_x_2}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[2]} {flag_for_x_2}')
        channel_index += 1
        second_row_s_circ_flagged.append_from_stim_program_text(f'TICK')
       
        # flag for z
        second_row_s_circ_flagged.append_from_stim_program_text(f'RX {flag_for_z}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'Z_ERROR[{gt(channel_index)}]({p}) {flag_for_z}')
        channel_index += 1
        second_row_s_circ_flagged.append_from_stim_program_text(f'TICK')
        second_row_s_circ_flagged.append_from_stim_program_text(f'CX {flag_for_z} {logical_idx[1]}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {flag_for_z} {logical_idx[1]}')
        channel_index += 1
        second_row_s_circ_flagged.append_from_stim_program_text(f'TICK')
        # logical CZ
        second_row_s_circ_flagged.append_from_stim_program_text(f'CZ {logical_idx[1]} {logical_idx[2]}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[1]} {logical_idx[2]}')
        channel_index += 1
        second_row_s_circ_flagged.append_from_stim_program_text(f'TICK')
        # flag for z close
        second_row_s_circ_flagged.append_from_stim_program_text(f'CZ {flag_for_z} {logical_idx[2]}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {flag_for_z} {logical_idx[2]}')
        channel_index += 1
        second_row_s_circ_flagged.append_from_stim_program_text(f'TICK')
        second_row_s_circ_flagged.append_from_stim_program_text(f'CX {flag_for_z} {logical_idx[1]}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {flag_for_z} {logical_idx[1]}')
        channel_index += 1
        second_row_s_circ_flagged.append_from_stim_program_text(f'TICK')
        second_row_s_circ_flagged.append_from_stim_program_text(f'Z_ERROR[{gt(channel_index)}]({p}) {flag_for_z}')
        channel_index += 1
        second_row_s_circ_flagged.append_from_stim_program_text(f'MX {flag_for_z}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'DETECTOR rec[-1]')
        # close int flag
        second_row_s_circ_flagged += circ_stab_2_b
        second_row_s_circ_flagged.append_from_stim_program_text(f'TICK')
         # flag for x close
        second_row_s_circ_flagged.append_from_stim_program_text(f'CX {logical_idx[2]} {flag_for_x_2}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'DEPOLARIZE2[{gt(channel_index)}]({p}) {logical_idx[2]} {flag_for_x_2}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'TICK')
        channel_index += 1
        second_row_s_circ_flagged.append_from_stim_program_text(f'X_ERROR[{gt(channel_index)}]({p}) {flag_for_x_2}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'M {flag_for_x_2}')
        second_row_s_circ_flagged.append_from_stim_program_text(f'DETECTOR rec[-1]')
        channel_index += 1
        second_row_s_circ_flagged.append_from_stim_program_text(f'TICK')
        # transversal S
        for i in range(len(logical_idx)):
                second_row_s_circ_flagged.append_from_stim_program_text(f'S {logical_idx[i]}')
                second_row_s_circ_flagged.append_from_stim_program_text(f'DEPOLARIZE1[{gt(channel_index)}]({p}) {logical_idx[i]}')
                channel_index += 1
        
        
        ## full round of error correction
        sm_circuit, first_free_idx, channel_index = stab_meas_circuit(p, code.hx, code.hz, first_free_index=first_free_idx, channel_index=channel_index)
        
        circuit_0 = circ_stab_1_a + circ_stab_1_b.without_noise() + sm_circuit.without_noise() + s_circ.without_noise()
        circuit_1 = circ_stab_1_a.without_noise()  + circ_stab_2_a  + circ_stab_1_b.without_noise()  + circ_stab_2_b.without_noise()  + sm_circuit.without_noise() + s_circ.without_noise()
        circuit_2 = circ_stab_1_a.without_noise() + circ_stab_2_a.without_noise() + first_row_s_circ  + second_row_s_circ_flagged + sm_circuit.without_noise()
        circuit_3 = circ_stab_1_a.without_noise() + circ_stab_2_a.without_noise() + first_row_s_circ.without_noise()  + second_row_s_circ_flagged.without_noise() + sm_circuit
        
        circuits = [circuit_0, circuit_1,circuit_2, circuit_3]
        
        luts = []
        for i_c,c in enumerate(circuits):
                ca = CircuitAnalyzer(c, code=code)

                lut = ca.construct_lut(verbose=False)
                luts += [lut]
        
        # for k,v in luts[2].items():
        #         print(k,v)
        
        cc_lut = load_lookup_table('./lut/1_3_cc.lut')
        
        # Then do protocol
        final_paulis_on_data = []
        circuit_branches = []
        all_dfs = []
        all_info_strs = []
        iterator = tqdm(range(nshots)) if nshots > 1 else range(nshots)
        for shot in iterator:
                info_strs = []
                
                simulator = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
                full_circuit = stim.Circuit()
                
                if unitary_encoding is not None:
                        simulator.do(unitary_encoding)
                        # full_circuit += unitary_encoding

                
                simulator.do(circ_stab_1_a)
                # full_circuit += circ_stab_1_a
                
                mf_1 = simulator.get_detector_flips(instance_index=0)
                pf_1 = simulator.peek_pauli_flips(instance_index=0)
                
                sm_1_outcome = mf_1[0]
                sf_1_outcome = mf_1[1]
                # info_strs += [f'sm_1 : {int(sm_1_outcome)}, sf_1 : {int(sf_1_outcome)}', f'{pf_1} {pf_1.pauli_indices()}']

                
                if np.any([sm_1_outcome, sf_1_outcome]): # 1 fault happened
                        # apply the error correction (for now: full error correction )
                        # info_strs += ['* fault detected in first stab meas - undo interface flag and do ec']
                        
                        simulator.do(circ_stab_1_b)
                        # full_circuit += circ_stab_1_b
                        
                        simulator.do(sm_circuit)
                        # full_circuit += sm_circuit
                        
                        pf_ec = simulator.peek_pauli_flips(instance_index=0)
                        mf_ec = simulator.get_detector_flips(instance_index=0)
                        
                        if_1_outcome = mf_ec[2]
                        s_outcome = mf_ec[-12:]
                        
                        # info_strs += [f'all dfs: {ppv(mf_ec)}' , f'synd part : {ppv(s_outcome)}', f'{pf_ec} {pf_ec.pauli_indices()}']
                        
                        # info_strs += ['* then do unprotected s gate']
                        simulator.do(s_circ)
                        # full_circuit += s_circ
                        pf_ec = simulator.peek_pauli_flips(instance_index=0)
                        
                        # info_strs += [f'{pf_ec} {pf_ec.pauli_indices()}']
                        
                        # apply the correction 
                        correction = luts[0].get(tuple(mf_ec.astype(int)), None)
                        if correction is None:
                                # info_strs += ['corr not found in flag lut, getting correction from cc lut ']
                                correction = stim.PauliString(cc_lut.get(tuple(s_outcome.astype(int)),'I'*15))
                        # info_strs += [f'corr :\n{correction} {correction.pauli_indices()}']
                        total_pf = pf_ec * correction
                        for i_p, p in enumerate(total_pf):
                                simulator.set_pauli_flip(pauli=p, qubit_index=i_p, instance_index=0)
                        
                        # info_strs += [f'totp :\n{total_pf} {total_pf.pauli_indices()}']
                        
                        
                        mf_ec = simulator.get_detector_flips(instance_index=0)
                        pf_ec = simulator.peek_pauli_flips(instance_index=0)
                        
                        # info_strs += [f'{pf_ec}']
                        circuit_branch =  0
                
                else: # no fault happened
                        # info_strs += ['* no fault detected in first stab meas - do second stabilizer']
                        simulator.do(circ_stab_2_a)
                        # full_circuit += circ_stab_2_a
                        
                        mf_2 = simulator.get_detector_flips(instance_index=0)
                        pf_2 = simulator.peek_pauli_flips(instance_index=0)
                        
                        sm_2_outcome = mf_2[-2]
                        sf_2_outcome = mf_2[-1]
                        # info_strs += [f'sm_2 : {int(sm_2_outcome)}, sf_2 : {int(sf_2_outcome)}', f'{pf_2} {pf_2.pauli_indices()}']
                        
                        if np.any([sm_2_outcome,sf_2_outcome]):
                                # info_strs += ['* fault detected in second stab meas- undo interface flags and do ec']
                                
                                simulator.do(circ_stab_1_b)
                                # full_circuit += circ_stab_1_b
                                
                                simulator.do(circ_stab_2_b)
                                # full_circuit += circ_stab_2_b
                                
                                simulator.do(sm_circuit)
                                # full_circuit += sm_circuit
                                
                                pf_ec = simulator.peek_pauli_flips(instance_index=0)
                                mf_ec = simulator.get_detector_flips(instance_index=0)

                                s_outcome = mf_ec[-12:]
                                
                                # info_strs += [f's : {ppv(s_outcome)}', f'{pf_ec} {pf_ec.pauli_indices()}']
                                
                                # info_strs += ['* then do unprotected s gate']
                                simulator.do(s_circ)
                                # full_circuit += s_circ
                                pf_ec = simulator.peek_pauli_flips(instance_index=0)
                                
                                # apply the correction 
                                correction = luts[1].get(tuple(mf_ec.astype(int)), None)
                                if correction is None:
                                        correction = stim.PauliString(cc_lut.get(tuple(s_outcome.astype(int)),'I'*15))
                                total_pf = pf_ec * correction
                                for i_p, p in enumerate(total_pf):
                                        simulator.set_pauli_flip(pauli=p, qubit_index=i_p, instance_index=0)
                                
                                
                                mf_ec = simulator.get_detector_flips(instance_index=0)
                                pf_ec = simulator.peek_pauli_flips(instance_index=0)
                                
                                # info_strs += [f'{pf_ec} {pf_ec.pauli_indices()}']
                                circuit_branch =  1
                                
                        
                        else:
                                # info_strs += ['* no fault detected in second stab meas - do S-gate with flags']
                                # info_strs += ['first row:']
                                
                                simulator.do(first_row_s_circ)
                                # full_circuit += first_row_s_circ
                                
                                mf_ec = simulator.get_detector_flips(instance_index=0)
                                ffz_1_1_outcome = mf_ec[-4]
                                fix_1_1_outcome = mf_ec[-3]
                                ffz_1_2_outcome = mf_ec[-2]
                                ffx_1_1_outcome = mf_ec[-1]
                                pf_ec = simulator.peek_pauli_flips(instance_index=0)
                                
                                # info_strs += [f'ffz_1_1 : {int(ffz_1_1_outcome)}, fix_1_1 : {int(fix_1_1_outcome)}, ffz_1_2 : {int(ffz_1_2_outcome)}, ffx_1_1 : {int(ffx_1_1_outcome)}', f'{pf_ec} {pf_ec.pauli_indices()}']
                                
                                
                                # info_strs += ['second row:']
                                simulator.do(second_row_s_circ_flagged)
                                # full_circuit += second_row_s_circ_flagged
                                
                                mf_ec = simulator.get_detector_flips(instance_index=0)
                                ffz_2_1_outcome = mf_ec[-4]
                                fix_2_1_outcome = mf_ec[-3]
                                fix_2_2_outcome = mf_ec[-2]
                                ffx_2_1_outcome = mf_ec[-1]
                                pf_ec = simulator.peek_pauli_flips(instance_index=0)
                                
                                # info_strs += [f'ffz_2_1 : {int(ffz_2_1_outcome)}, fix_2_1 : {int(fix_2_1_outcome)}, fix_2_2 : {int(fix_2_2_outcome)}, ffx_2_1 : {int(ffx_2_1_outcome)}', f'{pf_ec} {pf_ec.pauli_indices()}']
                                
                                
                                if np.any([ffz_1_1_outcome,fix_1_1_outcome,ffz_1_2_outcome,ffx_1_1_outcome,ffz_2_1_outcome,fix_2_1_outcome,fix_2_2_outcome,ffx_2_1_outcome]): # some fault happened here
                                        # info_strs += ['* fault detected in flagged s gate - do SM without flags']
                                        
                                        simulator.do(sm_circuit)
                                        # full_circuit += sm_circuit
                                        
                                        pf_ec = simulator.peek_pauli_flips(instance_index=0)
                                        mf_ec = simulator.get_detector_flips(instance_index=0)
                                        s_outcome = mf_ec[-12:]
                                        
                                        # info_strs += [f'df : {ppv(mf_ec)}',f's : {ppv(s_outcome)}', f'{pf_ec} {pf_ec.pauli_indices()}']
                                        # apply the correction 
                                        correction = luts[2].get(tuple(mf_ec.astype(int)), None)
                                        if correction is None:
                                                # info_strs += ['corr not found in flag lut, getting correction from cc lut ']
                                                correction = stim.PauliString(cc_lut.get(tuple(s_outcome.astype(int)),'I'*15))
                                        # info_strs += [f'corr :\n{correction} {correction.pauli_indices()}']
                                        total_pf = pf_ec * correction
                                        for i_p, p in enumerate(total_pf):
                                                simulator.set_pauli_flip(pauli=p, qubit_index=i_p, instance_index=0)
                                        
                                        # info_strs += [f'totp :\n{total_pf} {total_pf.pauli_indices()}']
                                        
                                        mf_ec = simulator.get_detector_flips(instance_index=0)
                                        pf_ec = simulator.peek_pauli_flips(instance_index=0)
                                        
                                        # info_strs += [f'{pf_ec} {pf_ec.pauli_indices()}']
                                        circuit_branch =  2
                                
                                else:
                                        # info_strs += ['* no fault detected in flagged s gate - should do FT SM with flags, for now doing nothing']
                                        
                                        
                                        # info_strs += ['* no fault detected in flagged s gate - should do FT SM with flags, for now doing noise-free round']
                                        
                                        simulator.do(sm_circuit.without_noise())
                                        # full_circuit += sm_circuit.without_noise()
                                        
                                        pf_ec = simulator.peek_pauli_flips(instance_index=0)
                                        mf_ec = simulator.get_detector_flips(instance_index=0)
                                        s_outcome = mf_ec[-12:]
                                        
                                        # apply the correction 
                                        # correction = luts[3].get(tuple(mf_ec.astype(int)), None)
                                        # if correction is None:
                                        correction = stim.PauliString(cc_lut.get(tuple(s_outcome.astype(int)),'I'*15))
                                        total_pf = pf_ec * correction
                                        for i_p, p in enumerate(total_pf):
                                                simulator.set_pauli_flip(pauli=p, qubit_index=i_p, instance_index=0)
                                        
                                        mf_ec = simulator.get_measurement_flips(instance_index=0)
                                        pf_ec = simulator.peek_pauli_flips(instance_index=0)
                                        
                                        # info_strs += [f'{pf_ec} {pf_ec.pauli_indices()}']
                                        circuit_branch =  3
                
                dfs = simulator.get_detector_flips(instance_index=0)
                all_dfs += [dfs]
                final_pauli_on_data = choose_indices_of_pauli_string(pf_ec, list(range(15)))
                # info_strs += ['final_pauli_on_data : ',f'{final_pauli_on_data}']
                
                # all_ info_strs += [info_strs]
                
                final_paulis_on_data += [final_pauli_on_data]
                circuit_branches += [circuit_branch]
        
        if nshots == 1:
                return final_pauli_on_data, full_circuit, '\n'.join(info_strs), circuit_branch, circuits
        else:
                return final_paulis_on_data, circuit_branches, all_dfs, all_info_strs