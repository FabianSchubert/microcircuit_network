import matplotlib.pyplot as plt
import numpy as np

from mc.network import Network

from tests import test_model_rate_spikes as net_model

from ..utils import plot_spike_times, calc_loss_interp

col_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

import pdb
import ipdb

import mnist

######################################################
# adjust model parameters

#net_model.synapses.IP.mod_dat["wu_param_space_plast"]["muIP"] = 1e-4
#net_model.synapses.PI.mod_dat["wu_param_space_plast"]["muPI"] = 3e-5
#net_model.synapses.PINP.mod_dat["wu_param_space_plast"]["muPINP"] = 1e-3
#net_model.synapses.PPBasal.mod_dat["wu_param_space_plast"]["muPP_basal"] = 1e-4
######################################################

######################################################
# network parameters
N_IN = 784
N_HIDDEN = [500]
N_OUT = 10

DT = 0.2
######################################################

######################################################
# simulation run parameters
T = 60000
NT = int(T / DT)
T = NT * DT

T_VAL = 30000
NT_VAL = int(T_VAL / DT)
T_VAL = NT_VAL * DT

T_SHOW_PATTERNS = 200
NT_SHOW_PATTERNS = int(T_SHOW_PATTERNS / DT)
T_SHOW_PATTERNS = NT_SHOW_PATTERNS * DT

T_IND_UPDATE_PATTERNS = np.arange(NT)[::NT_SHOW_PATTERNS]
N_UPDATE_PATTERNS = T_IND_UPDATE_PATTERNS.shape[0]

T_IND_UPDATE_PATTERNS_VAL = np.arange(NT_VAL)[::NT_SHOW_PATTERNS]
N_UPDATE_PATTERNS_VAL = T_IND_UPDATE_PATTERNS_VAL.shape[0]

NT_SKIP_REC = 600
T_IND_REC = np.arange(NT)[::NT_SKIP_REC]
N_REC = T_IND_REC.shape[0]

NT_SKIP_REC_VAL = 10
T_IND_REC_VAL = np.arange(NT_VAL)[::NT_SKIP_REC_VAL]
N_REC_VAL = T_IND_REC_VAL.shape[0]

N_VAL_RUNS = 20
T_IND_VAL_RUNS = np.linspace(0., NT - 1, N_VAL_RUNS).astype("int")
######################################################

######################################################
# generate some training data
train_images = np.array(2.0 * mnist.train_images()/255.)
N_TRAIN_SAMPLES = train_images.shape[0]
train_labels = np.array(mnist.train_labels()).astype("int")

# no epochs, just a sequence of random indices for the training samples without
# constraints regarding redundancy.
# (e.g. this does not guarantee that all training samples will be shown before
# an image is presented for the second, third, etc. time )
ind_sample_train = np.random.randint(0, N_TRAIN_SAMPLES, N_UPDATE_PATTERNS)

INPUT_DATA = np.reshape(train_images[ind_sample_train], (N_UPDATE_PATTERNS, N_IN))
INPUT_DATA_FLAT = INPUT_DATA.flatten()

# V_HIDDEN = (W_10 @ phi(INPUT_DATA.T)).T
# R_HIDDEN = phi(V_HIDDEN)

OUTPUT_DATA = np.zeros((N_UPDATE_PATTERNS, N_OUT))
OUTPUT_DATA[np.arange(N_UPDATE_PATTERNS), train_labels[ind_sample_train]] = 2.0
OUTPUT_DATA_FLAT = OUTPUT_DATA.flatten()
######################################################

######################################################
# generate validation data

val_images = 2.0 * mnist.test_images()/255.
N_VAL_SAMPLES = val_images.shape[0]
val_labels = mnist.test_labels()

ind_sample_val = np.random.randint(0, N_VAL_SAMPLES, N_UPDATE_PATTERNS_VAL)

# generate a single validation example
INPUT_DATA_VAL = np.reshape(val_images[ind_sample_val], (N_UPDATE_PATTERNS_VAL, N_IN))
INPUT_DATA_VAL_FLAT = INPUT_DATA_VAL.flatten()

OUTPUT_DATA_VAL = np.zeros((N_UPDATE_PATTERNS_VAL, N_OUT))
OUTPUT_DATA_VAL[np.arange(N_UPDATE_PATTERNS_VAL), val_labels[ind_sample_val]] = 2.0
OUTPUT_DATA_VAL_FLAT = OUTPUT_DATA_VAL.flatten()

NEUR_READOUT_VAL = [("neur_input_input_pop", "r", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "vb", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "vbEff", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "va", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "u", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "r", T_IND_REC_VAL),
                    ("neur_hidden0_int_pop", "v", T_IND_REC_VAL),
                    ("neur_hidden0_int_pop", "vEff", T_IND_REC_VAL),
                    ("neur_hidden0_int_pop", "u", T_IND_REC_VAL),
                    ("neur_hidden0_int_pop", "r", T_IND_REC_VAL),
                    ("neur_output_output_pop", "vb", T_IND_REC_VAL),
                    ("neur_output_output_pop", "vbEff", T_IND_REC_VAL),
                    ("neur_output_output_pop", "vnudge", T_IND_REC_VAL),
                    ("neur_output_output_pop", "u", T_IND_REC_VAL),
                    ("neur_output_output_pop", "r", T_IND_REC_VAL)]

DICT_DATA_VALIDATION = {"T": NT_VAL,
                        "t_sign": T_IND_UPDATE_PATTERNS_VAL,
                        "ext_data_input": INPUT_DATA_VAL_FLAT,
                        "ext_data_pop_vars": [
                            (np.zeros((1, N_OUT)),
                             np.array([0.]).astype("int"),
                             "neur_output_output_pop", "gnudge")],
                        "readout_neur_pop_vars": NEUR_READOUT_VAL}

# copy it for each validation
DATA_VALIDATION = [DICT_DATA_VALIDATION] * N_VAL_RUNS

NEUR_POPS_SPIKE_REC_VAL = ["neur_input_input_pop",
                           "neur_hidden0_pyr_pop",
                           "neur_hidden0_int_pop",
                           "neur_output_output_pop"]
######################################################

######################################################
# recording settings
NEUR_POPS_SPIKE_REC = []

'''["neur_input_input_pop",
                       "neur_hidden0_pyr_pop",
                       "neur_hidden0_int_pop",
                       "neur_output_output_pop"]'''

NEUR_READOUT = [("neur_input_input_pop", "r", T_IND_REC),
                ("neur_hidden0_pyr_pop", "vb", T_IND_REC),
                ("neur_hidden0_pyr_pop", "vbEff", T_IND_REC),
                ("neur_hidden0_pyr_pop", "va", T_IND_REC),
                ("neur_hidden0_pyr_pop", "u", T_IND_REC),
                ("neur_hidden0_pyr_pop", "r", T_IND_REC),
                ("neur_hidden0_int_pop", "v", T_IND_REC),
                ("neur_hidden0_int_pop", "vEff", T_IND_REC),
                ("neur_hidden0_int_pop", "u", T_IND_REC),
                ("neur_hidden0_int_pop", "r", T_IND_REC),
                ("neur_output_output_pop", "vb", T_IND_REC),
                ("neur_output_output_pop", "vbEff", T_IND_REC),
                ("neur_output_output_pop", "vnudge", T_IND_REC),
                ("neur_output_output_pop", "u", T_IND_REC),
                ("neur_output_output_pop", "r", T_IND_REC)]

SYN_READOUT = [("syn_input_input_pop_to_hidden0_pyr_pop",
                "g", T_IND_REC),
               ("syn_hidden0_pyr_pop_to_int_pop",
                "g", T_IND_REC),
               ("syn_hidden0_int_pop_to_pyr_pop",
                "g", T_IND_REC),
               ("syn_hidden0_pyr_pop_to_output_output_pop",
                "g", T_IND_REC),
               ("syn_output_output_pop_to_hidden0_pyr_pop",
                "g", T_IND_REC)]
######################################################

######################################################
'''
name: str
model_def: types.ModuleType
size_input: int
size_hidden: list
size_output: int
t_inp_max: int
spike_buffer_size: int
spike_buffer_size_val: int
spike_rec_pops: list = field(default_factory=list)
spike_rec_pops_val: list = field(default_factory=list)
dt: float = 0.1
plastic: bool = True
t_inp_static_max: int = 0
'''
# Initialize network
net = Network("testnet",
              net_model,
              N_IN, N_HIDDEN, N_OUT,
              N_UPDATE_PATTERNS,
              NT,
              NT_VAL,
              dt=DT,
              spike_rec_pops=NEUR_POPS_SPIKE_REC,
              spike_rec_pops_val=NEUR_POPS_SPIKE_REC_VAL,
              plastic=True,
              t_inp_static_max=N_UPDATE_PATTERNS_VAL)

# set the nudging conductance in the output layer
# neurons to zero (we do not want to drive the output here).
# net.neur_pops["neur_output_output_pop"].vars["gnudge"].view[:] = 0.0
# net.neur_pops["neur_output_output_pop"].push_var_to_device("gnudge")

# set the IP and PI weights into the self-predicting state.
net.init_self_pred_state()
######################################################

######################################################
# run sim
(results_neur, results_syn,
 results_spike, results_validation) = net.run_sim(NT,
                                                  T_IND_UPDATE_PATTERNS,
                                                  INPUT_DATA_FLAT, OUTPUT_DATA_FLAT,
                                                  None,
                                                  NEUR_READOUT,
                                                  SYN_READOUT,
                                                  T_IND_VAL_RUNS,
                                                  DATA_VALIDATION)
######################################################

######################################################
# result vars
inp_r = results_neur["neur_input_input_pop_r"]
#inp_sp = results_spike["neur_input_input_pop"]

p_vb = results_neur["neur_hidden0_pyr_pop_vb"]
p_vbeff = results_neur["neur_hidden0_pyr_pop_vbEff"]
p_va = results_neur["neur_hidden0_pyr_pop_va"]
p_u = results_neur["neur_hidden0_pyr_pop_u"]
p_r = results_neur["neur_hidden0_pyr_pop_r"]
#p_sp = results_spike["neur_hidden0_pyr_pop"]

i_v = results_neur["neur_hidden0_int_pop_v"]
i_veff = results_neur["neur_hidden0_int_pop_vEff"]
i_u = results_neur["neur_hidden0_int_pop_u"]
i_r = results_neur["neur_hidden0_int_pop_r"]
#i_sp = results_spike["neur_hidden0_int_pop"]

out_vb = results_neur["neur_output_output_pop_vb"]
out_vbeff = results_neur["neur_output_output_pop_vbEff"]
out_u = results_neur["neur_output_output_pop_u"]
out_r = results_neur["neur_output_output_pop_r"]
#out_sp = results_spike["neur_output_output_pop"]
######################################################

loss = np.ndarray((N_VAL_RUNS))

for k in range(N_VAL_RUNS):

    _u_readout = results_validation[k]["neur_var_rec"]["neur_output_output_pop_u"]

    loss[k] = calc_loss_interp(T_IND_UPDATE_PATTERNS_VAL,
                               T_IND_REC_VAL,
                               OUTPUT_DATA_VAL, _u_readout,
                               perc_readout_targ_change=0.9)

plt.ion()

fig_pred, ax_pred = plt.subplots(1,1)
ax_pred.step(T_IND_UPDATE_PATTERNS_VAL,OUTPUT_DATA_VAL[:,0],where="post")
ax_pred.plot(T_IND_REC_VAL,results_validation[-1]["neur_var_rec"]["neur_output_output_pop_u"][:,0])

fig_w, ax_w = plt.subplots(1,1)

for readout in SYN_READOUT:
    _wrec = results_syn[f'{readout[0]}_g'][::10,0,0]
    ax_w.plot(_wrec, label = readout[0])

ax_w.legend()


pdb.set_trace()