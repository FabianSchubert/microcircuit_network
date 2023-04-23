"""
This test runs a microcircuit model
with one hidden layer on input data
that consists of randomly generated
patterns that are shown over a certain
period of time.
"""

import matplotlib.pyplot as plt

import numpy as np

import pickle

from mc.network import Network

from models import test_model_yinyang_integrator_spikes as net_model

from ..utils import plot_spike_times, calc_loss_interp

from .utils import gen_output_data

col_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

import pdb

import os

import tikzplotlib

BASE_FOLD = os.path.dirname(__import__(__name__).__file__)


######################################################
# network parameters
N_IN = 2
N_HIDDEN = [30]
N_OUT = 4

N_BATCH = 128
N_BATCH_VAL = 10

DT = 0.2
######################################################

######################################################
# simulation run parameters
T = 500000
NT = int(T / DT)
T = NT * DT

T_SKIP_BATCH_PLAST = 100

T_VAL = 30000
NT_VAL = int(T_VAL / DT)
T_VAL = NT_VAL * DT

T_SHOW_PATTERNS = 800
NT_SHOW_PATTERNS = int(T_SHOW_PATTERNS / DT)
T_SHOW_PATTERNS = NT_SHOW_PATTERNS * DT

T_IND_UPDATE_PATTERNS = np.arange(NT)[::NT_SHOW_PATTERNS]
N_UPDATE_PATTERNS = T_IND_UPDATE_PATTERNS.shape[0]

T_IND_UPDATE_PATTERNS_VAL = np.arange(NT_VAL)[::NT_SHOW_PATTERNS]
N_UPDATE_PATTERNS_VAL = T_IND_UPDATE_PATTERNS_VAL.shape[0]

NT_SKIP_REC = 5000
T_IND_REC = np.arange(NT)[::NT_SKIP_REC]
N_REC = T_IND_REC.shape[0]

NT_SKIP_REC_VAL = 1000
T_IND_REC_VAL = np.arange(NT_VAL)[::NT_SKIP_REC_VAL]
N_REC_VAL = T_IND_REC_VAL.shape[0]

N_VAL_RUNS = 10
T_IND_VAL_RUNS = np.linspace(0., NT - 1, N_VAL_RUNS).astype("int")
######################################################

######################################################
# generate some training data
# random voltage values -> if smaller zero, no output
U_MAX = 1.0
U_MIN = 0.0

INPUT_DATA = np.random.rand(N_UPDATE_PATTERNS, N_BATCH, N_IN) * (U_MAX - U_MIN) + U_MIN
INPUT_DATA_FLAT = INPUT_DATA.flatten()

OUTPUT_DATA = gen_output_data(INPUT_DATA)

OUTPUT_DATA_FLAT = OUTPUT_DATA.flatten()
######################################################

######################################################
# generate validation data

# generate a single validation example
INPUT_DATA_VAL = np.random.rand(N_UPDATE_PATTERNS_VAL, N_BATCH_VAL, N_IN) * (U_MAX - U_MIN) + U_MIN
INPUT_DATA_VAL_FLAT = INPUT_DATA_VAL.flatten()

OUTPUT_DATA_VAL = gen_output_data(INPUT_DATA_VAL)

OUTPUT_DATA_VAL_FLAT = OUTPUT_DATA_VAL.flatten()

NEUR_READOUT_VAL = [("neur_input_input_pop", "r", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "vb", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "vbEff", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "va", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "u", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "rEff", T_IND_REC_VAL),
                    ("neur_hidden0_pyr_pop", "ca", T_IND_REC_VAL),
                    #("neur_hidden0_pyr_pop", "r", T_IND_REC_VAL),
                    ("neur_hidden0_int_pop", "v", T_IND_REC_VAL),
                    ("neur_hidden0_int_pop", "vEff", T_IND_REC_VAL),
                    ("neur_hidden0_int_pop", "u", T_IND_REC_VAL),
                    ("neur_hidden0_int_pop", "rEff", T_IND_REC_VAL),
                    ("neur_hidden0_int_pop", "ca", T_IND_REC_VAL),
                    #("neur_hidden0_int_pop", "r", T_IND_REC_VAL),
                    ("neur_output_output_pop", "vb", T_IND_REC_VAL),
                    ("neur_output_output_pop", "vbEff", T_IND_REC_VAL),
                    ("neur_output_output_pop", "vnudge", T_IND_REC_VAL),
                    ("neur_output_output_pop", "u", T_IND_REC_VAL),
                    ("neur_output_output_pop", "rEff", T_IND_REC_VAL),
                    ("neur_output_output_pop", "ca", T_IND_REC_VAL)]#,
                    #("neur_output_output_pop", "r", T_IND_REC_VAL)]

DICT_DATA_VALIDATION = {"T": NT_VAL,
                        "t_sign": T_IND_UPDATE_PATTERNS_VAL,
                        "ext_data_input": INPUT_DATA_VAL_FLAT,
                        "ext_data_pop_vars": [
                            (np.zeros((1, N_BATCH_VAL, N_OUT)),
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
                ("neur_hidden0_pyr_pop", "va_int", T_IND_REC),
                ("neur_hidden0_pyr_pop", "va_exc", T_IND_REC),
                ("neur_hidden0_pyr_pop", "u", T_IND_REC),
                ("neur_hidden0_pyr_pop", "rEff", T_IND_REC),
                ("neur_hidden0_pyr_pop", "ca", T_IND_REC),
                ("neur_hidden0_pyr_pop", "I_bias", T_IND_REC),
                #("neur_hidden0_pyr_pop", "r", T_IND_REC),
                ("neur_hidden0_int_pop", "v", T_IND_REC),
                ("neur_hidden0_int_pop", "vEff", T_IND_REC),
                ("neur_hidden0_int_pop", "u", T_IND_REC),
                ("neur_hidden0_int_pop", "rEff", T_IND_REC),
                ("neur_hidden0_int_pop", "ca", T_IND_REC),
                ("neur_hidden0_int_pop", "I_bias", T_IND_REC),
                #("neur_hidden0_int_pop", "r", T_IND_REC),
                ("neur_output_output_pop", "vb", T_IND_REC),
                ("neur_output_output_pop", "vbEff", T_IND_REC),
                ("neur_output_output_pop", "vnudge", T_IND_REC),
                ("neur_output_output_pop", "u", T_IND_REC),
                ("neur_output_output_pop", "rEff", T_IND_REC),
                ("neur_output_output_pop", "ca", T_IND_REC)]#,
                #("neur_output_output_pop", "r", T_IND_REC)]

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
              t_inp_static_max=N_UPDATE_PATTERNS_VAL,
              n_batches=N_BATCH,
              n_batches_val=N_BATCH_VAL)

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
                                                  DATA_VALIDATION,
                                                  T_skip_batch_plast=T_SKIP_BATCH_PLAST)
######################################################

######################################################
# result vars
inp_r = results_neur["neur_input_input_pop_r"]
#inp_sp = results_spike["neur_input_input_pop"]

p_vb = results_neur["neur_hidden0_pyr_pop_vb"]
p_vbeff = results_neur["neur_hidden0_pyr_pop_vbEff"]
p_va = results_neur["neur_hidden0_pyr_pop_va"]
p_va_int = results_neur["neur_hidden0_pyr_pop_va_int"]
p_va_exc = results_neur["neur_hidden0_pyr_pop_va_exc"]
p_u = results_neur["neur_hidden0_pyr_pop_u"]
#p_r = results_neur["neur_hidden0_pyr_pop_r"]
#p_sp = results_spike["neur_hidden0_pyr_pop"]
p_reff = results_neur["neur_hidden0_pyr_pop_rEff"]
p_ca = results_neur["neur_hidden0_pyr_pop_ca"]
p_bias = results_neur["neur_hidden0_pyr_pop_I_bias"]

i_v = results_neur["neur_hidden0_int_pop_v"]
i_veff = results_neur["neur_hidden0_int_pop_vEff"]
i_u = results_neur["neur_hidden0_int_pop_u"]
#i_r = results_neur["neur_hidden0_int_pop_r"]
#i_sp = results_spike["neur_hidden0_int_pop"]
i_reff = results_neur["neur_hidden0_int_pop_rEff"]
i_ca = results_neur["neur_hidden0_int_pop_ca"]
i_bias = results_neur["neur_hidden0_int_pop_I_bias"]

out_vb = results_neur["neur_output_output_pop_vb"]
out_vbeff = results_neur["neur_output_output_pop_vbEff"]
out_u = results_neur["neur_output_output_pop_u"]
#out_r = results_neur["neur_output_output_pop_r"]
#out_sp = results_spike["neur_output_output_pop"]
out_reff = results_neur["neur_output_output_pop_rEff"]
out_ca = results_neur["neur_output_output_pop_ca"]
######################################################

loss = np.ndarray((N_VAL_RUNS))

for k in range(N_VAL_RUNS):

    _ca_readout = results_validation[k]["neur_var_rec"]["neur_output_output_pop_ca"]

    loss[k] = calc_loss_interp(T_IND_UPDATE_PATTERNS_VAL,
                               T_IND_REC_VAL,
                               OUTPUT_DATA_VAL, _ca_readout,
                               perc_readout_targ_change=0.9)

plt.ion()

######################################################
fig_loss, ax_loss = plt.subplots(1, 1)

ax_loss.plot(T_IND_VAL_RUNS*DT, loss, '-o')
ax_loss.set_yscale("log")

ax_loss.set_ylabel("MSE")
ax_loss.set_xlabel(r'$t$')
######################################################

######################################################
fig_pred, ax_pred = plt.subplots(1,1)
ax_pred.step(T_IND_UPDATE_PATTERNS_VAL*DT,OUTPUT_DATA_VAL[:,0,0],where="post")
ax_pred.plot(T_IND_REC_VAL*DT,results_validation[-1]["neur_var_rec"]["neur_output_output_pop_ca"][:,0,0])
ax_pred.set_xlabel(r'$t$')
ax_pred.set_ylabel("Output Rate")
######################################################

######################################################
fig_w, ax_w = plt.subplots(1,1)

for readout in SYN_READOUT:
    _wrec = np.linalg.norm(results_syn[f'{readout[0]}_g'], axis=(1,2))
    ax_w.plot(_wrec, label = readout[0])

ax_w.legend()
######################################################

######################################################
PLOT_FOLD = os.path.join(BASE_FOLD, "plots")


tikzplotlib.save(os.path.join(PLOT_FOLD, "output_voltage.tex"), figure=fig_pred,
                 axis_width='\\figVoltW', axis_height='\\figVoltH')

tikzplotlib.save(os.path.join(PLOT_FOLD, "weights.tex"), figure=fig_w,
                 axis_width='\\figWeightsW', axis_height='\\figWeightsH')

tikzplotlib.save(os.path.join(PLOT_FOLD, "loss.tex"), figure=fig_loss,
                 axis_width='\\figLossW', axis_height='\\figLossH')

pickle.dump(fig_pred, open(os.path.join(PLOT_FOLD, "output_voltage.pickle"), "wb"))
pickle.dump(fig_w,  open(os.path.join(PLOT_FOLD, "weights.pickle"), "wb"))
pickle.dump(fig_loss, open(os.path.join(PLOT_FOLD, "loss.pickle"), "wb"))

fig_pred.savefig(os.path.join(PLOT_FOLD, "output_voltage.png"), dpi=600)
fig_w.savefig(os.path.join(PLOT_FOLD, "weights.png"), dpi=600)
fig_loss.savefig(os.path.join(PLOT_FOLD, "loss.png"), dpi=600)
######################################################

pdb.set_trace()
