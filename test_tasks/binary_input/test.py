"""
This test runs a microcircuit model
with one hidden layer on input data
that consists of randomly generated
patterns that are shown over a certain
period of time.
"""

import matplotlib.pyplot as plt
import numpy as np

from mc.network import Network

from tests import test_model_rate_spikes as net_model

from ..utils import plot_spike_times

col_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

######################################################
# network parameters
N_IN = 30
N_HIDDEN = [20]
N_OUT = 10

DT = 0.1
######################################################

######################################################
# simulation run parameters
T = 5000
NT = int(T/DT)
T = NT * DT

T_SHOW_PATTERNS = 200
NT_SHOW_PATTERNS = int(T_SHOW_PATTERNS/DT)
T_SHOW_PATTERNS = NT_SHOW_PATTERNS * DT

T_IND_UPDATE_PATTERNS = np.arange(NT)[::NT_SHOW_PATTERNS]
N_UPDATE_PATTERNS = T_IND_UPDATE_PATTERNS.shape[0]

NT_SKIP_REC = 10
T_IND_REC = np.arange(NT)[::NT_SKIP_REC]
N_REC = T_IND_REC.shape[0]
######################################################

######################################################
# generate some input data
# random voltage values -> if smaller zero, no output
U_MAX = 1.0
U_MIN = -3.0

INPUT_DATA = np.random.rand(N_UPDATE_PATTERNS, N_IN) * (U_MAX - U_MIN) + U_MIN
INPUT_DATA_FLAT = INPUT_DATA.flatten()
######################################################

######################################################
# recording settings
NEUR_POPS_SPIKE_REC = ["neur_input_input_pop",
                       "neur_hidden0_pyr_pop",
                       "neur_hidden0_int_pop",
                       "neur_output_output_pop"]

NEUR_READOUT = [("neur_input_input_pop", "r", T_IND_REC),
                ("neur_hidden0_pyr_pop", "vb", T_IND_REC),
                ("neur_hidden0_pyr_pop", "va", T_IND_REC),
                ("neur_hidden0_pyr_pop", "u", T_IND_REC),
                ("neur_hidden0_pyr_pop", "r", T_IND_REC),
                ("neur_hidden0_int_pop", "v", T_IND_REC),
                ("neur_hidden0_int_pop", "u", T_IND_REC),
                ("neur_hidden0_int_pop", "r", T_IND_REC),
                ("neur_output_output_pop", "vb", T_IND_REC),
                ("neur_output_output_pop", "vnudge", T_IND_REC),
                ("neur_output_output_pop", "u", T_IND_REC),
                ("neur_output_output_pop", "r", T_IND_REC)]

SYN_READOUT = None
######################################################

######################################################
# Initialize network
net = Network("testnet",
              net_model,
              N_IN, N_HIDDEN, N_OUT,
              N_UPDATE_PATTERNS,
              NT,
              0,
              dt=DT,
              spike_rec_pops=NEUR_POPS_SPIKE_REC,
              spike_rec_pops_val=[],
              plastic=False, t_inp_static_max=NT)

# set the nudging conductance in the output layer
# neurons to zero (we do not want to drive the output here).
net.neur_pops["neur_output_output_pop"].vars["gnudge"].view[:] = 0.0
net.neur_pops["neur_output_output_pop"].push_var_to_device("gnudge")
######################################################

# get the weights

######################################################
# run sim
(results_neur, results_syn,
 results_spike) = net.run_sim(NT,
                              T_IND_UPDATE_PATTERNS,
                              INPUT_DATA_FLAT, None,
                              None,
                              NEUR_READOUT,
                              SYN_READOUT)
######################################################

######################################################
# result vars
inp_r = results_neur["neur_input_input_pop_r"]
inp_sp = results_spike["neur_input_input_pop"]

p_vb = results_neur["neur_hidden0_pyr_pop_vb"]
p_va = results_neur["neur_hidden0_pyr_pop_va"]
p_u = results_neur["neur_hidden0_pyr_pop_u"]
p_r = results_neur["neur_hidden0_pyr_pop_r"]
p_sp = results_spike["neur_hidden0_pyr_pop"]

i_v = results_neur["neur_hidden0_int_pop_v"]
i_u = results_neur["neur_hidden0_int_pop_u"]
i_r = results_neur["neur_hidden0_int_pop_r"]
i_sp = results_spike["neur_hidden0_int_pop"]

out_vb = results_neur["neur_output_output_pop_vb"]
out_u = results_neur["neur_output_output_pop_u"]
out_r = results_neur["neur_output_output_pop_r"]
out_sp = results_spike["neur_output_output_pop"]
######################################################

import pdb
plt.ion()
pdb.set_trace()