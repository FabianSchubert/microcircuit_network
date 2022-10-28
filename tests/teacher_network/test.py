#! /usr/bin/env python3

import ipdb
import matplotlib.pyplot as plt
import numpy as np

import pdb

from mc.rate_mc.network import Network

from mc.rate_mc.neurons.params import (pyr_hidden_param_space,
                                       output_param_space,
                                       int_param_space)

from .utils import gen_input_output_data

gb_pyr = pyr_hidden_param_space["gb"]
ga_pyr = pyr_hidden_param_space["ga"]
glk_pyr = pyr_hidden_param_space["glk"]

gb_out = output_param_space["gb"]
ga_out = output_param_space["ga"]
glk_out = output_param_space["glk"]

glk_int = int_param_space["glk"]
gd_int = int_param_space["gd"]
gsom_int = int_param_space["gsom"]

#####################
# Training parameters
N_IN = 30
N_HIDDEN = [50]
N_OUT = 10

N_HIDDEN_TEACHER = 20

T_SHOW_PATTERNS = 150
N_PATTERNS = 10000

T_OFFSET = 0

T = N_PATTERNS * T_SHOW_PATTERNS + T_OFFSET
####################

######################
# recording parameters

T_SKIP_REC = 25
T_OFFSET_REC = 140
T_AX_READOUT = np.arange(T)[::T_SKIP_REC]+T_OFFSET_REC
######################

#####################
# plotting parameters

# skip readout data
T_SKIP_PLOT = 1
#####################

#######################
# validation parameters

T_INTERVAL_VALIDATION = int(T/10)

N_VAL_PATTERNS = 20

T_RUN_VALIDATION = T_SHOW_PATTERNS * N_VAL_PATTERNS

T_SIGN_VALIDATION = np.arange(T)[::T_INTERVAL_VALIDATION]
N_VALIDATION = T_SIGN_VALIDATION.shape[0]

T_SKIP_REC_VAL = 10
T_OFFSET_REC_VAL = 0
T_AX_READOUT_VAL = np.arange(T_RUN_VALIDATION)[::T_SKIP_REC_VAL] \
                        +T_OFFSET_REC_VAL
#######################

########################
# generate training data
(t_ax_train, test_input,
 test_output, W_10, W_21) = gen_input_output_data(N_IN,
                                                  N_HIDDEN_TEACHER,
                                                  N_OUT,
                                                  N_PATTERNS,
                                                  T_SHOW_PATTERNS,
                                                  T_OFFSET)
# No extra variable modifications
# throughout the simulation.
ext_data_pop_vars = []
########################

##########################
# generate validation data
data_validation = []

targ_output_validation = []

for k in range(N_VALIDATION):

    (t_ax_val, val_input,
     val_output, _, _) = gen_input_output_data(N_IN,
                                               N_HIDDEN_TEACHER,
                                               N_OUT,
                                               N_VAL_PATTERNS,
                                               T_SHOW_PATTERNS,
                                               0,
                                               (W_10, W_21))

    #{T, t_sign, ext_data_input, ext_data_pop_vars, readout_neur_pop_vars}

    _dict_data_validation = {}

    _dict_data_validation["T"] = T_RUN_VALIDATION
    _dict_data_validation["t_sign"] = t_ax_val
    _dict_data_validation["ext_data_input"] = val_input
    _dict_data_validation["ext_data_pop_vars"] = [(np.zeros((1, N_OUT)), 
                        np.array([0.]).astype("int"),
                       "neur_output_output_pop", "gnudge")]
    # This sets the nudging conductance of the output to zero
    # at the beginning of the evaluation.
    
    _dict_data_validation["readout_neur_pop_vars"] = [("neur_output_output_pop",
                                                    "u", T_AX_READOUT_VAL)]

    data_validation.append(_dict_data_validation)

    targ_output_validation.append(val_output)

##########################

# Initialize network
'''
name: str
size_input: int
size_hidden: list
size_output: int
t_inp_max: int
dt: float = 0.1
plastic: bool = True
t_inp_static_max: int = 0
'''

net = Network("testnet", N_IN, N_HIDDEN, N_OUT,
              N_PATTERNS, dt=0.5, plastic=True,
              t_inp_static_max=N_VAL_PATTERNS)


#################################
# Manually initialize the network
# in the self-predicting state

synview_pi = net.syn_pops["syn_hidden0_int_pop_to_pyr_pop"].vars["g"].view

synview_pp_back = net.syn_pops["syn_output_output_pop_to_hidden0_pyr_pop"
                               ].vars["g"].view

net.syn_pops["syn_output_output_pop_to_hidden0_pyr_pop"
             ].pull_var_from_device("g")

synview_pi[:] = -synview_pp_back

net.syn_pops["syn_hidden0_int_pop_to_pyr_pop"].push_var_to_device("g")

synview_ip = net.syn_pops["syn_hidden0_pyr_pop_to_int_pop"].vars["g"].view

synview_pp_fwd = net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"
                              ].vars["g"].view

net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"
             ].pull_var_from_device("g")

synview_ip = synview_pp_fwd * (gb_pyr + glk_pyr) / (gb_pyr + ga_pyr + glk_pyr)

net.syn_pops["syn_hidden0_pyr_pop_to_int_pop"].push_var_to_device("g")

#################################

##################################
# prepare the readout instructions
neur_readout_list = [("neur_output_output_pop",
                     "vb", T_AX_READOUT),
                     ("neur_output_output_pop",
                     "vnudge", T_AX_READOUT),
                     ("neur_output_output_pop",
                      "u", T_AX_READOUT),
                     ("neur_output_output_pop",
                      "r", T_AX_READOUT),
                     ("neur_hidden0_pyr_pop",
                      "u", T_AX_READOUT),
                     ("neur_hidden0_pyr_pop",
                      "va_exc", T_AX_READOUT),
                     ("neur_hidden0_pyr_pop",
                      "va_int", T_AX_READOUT),
                     ("neur_hidden0_pyr_pop",
                      "vb", T_AX_READOUT),
                     ("neur_hidden0_pyr_pop",
                      "va", T_AX_READOUT),
                     ("neur_hidden0_int_pop",
                      "r", T_AX_READOUT),
                     ("neur_hidden0_int_pop",
                      "u", T_AX_READOUT),
                     ("neur_hidden0_int_pop",
                      "v", T_AX_READOUT),
                     ("neur_hidden0_pyr_pop",
                      "r", T_AX_READOUT),
                     ("neur_input_input_pop",
                        "r", T_AX_READOUT)]

syn_readout_list = [("syn_hidden0_int_pop_to_pyr_pop",
                    "g", T_AX_READOUT),
                    ("syn_hidden0_pyr_pop_to_int_pop",
                    "g", T_AX_READOUT),
                    ("syn_hidden0_pyr_pop_to_output_output_pop",
                    "g", T_AX_READOUT),
                    ("syn_output_output_pop_to_hidden0_pyr_pop",
                    "g", T_AX_READOUT),
                    ("syn_input_input_pop_to_hidden0_pyr_pop",
                    "g", T_AX_READOUT)]
'''
T, t_sign, ext_data_input, ext_data_output,
ext_data_pop_vars, readout_neur_pop_vars,
readout_syn_pop_vars, t_sign_validation=None, data_validation=None

T, t_ax_train, test_input, test_output,
ext_data_pop_vars, neur_readout_list, syn_readout_list, t_sign_validation,
data_validation
'''

(results_neur, results_syn,
 results_validation) = net.run_sim(T, t_ax_train,
                                   test_input, test_output,
                                   ext_data_pop_vars,
                                   neur_readout_list,
                                   syn_readout_list,
                                   T_SIGN_VALIDATION,
                                   data_validation)

va_exc_hidden = results_neur["neur_hidden0_pyr_pop_va_exc"]
va_int_hidden = results_neur["neur_hidden0_pyr_pop_va_int"]
va_hidden = results_neur["neur_hidden0_pyr_pop_va"]
vb_hidden = results_neur["neur_hidden0_pyr_pop_vb"]
u_hidden = results_neur["neur_hidden0_pyr_pop_u"]
r_int_hidden = results_neur["neur_hidden0_int_pop_r"]
r_pyr_hidden = results_neur["neur_hidden0_pyr_pop_r"]

u_int_hidden = results_neur["neur_hidden0_int_pop_u"]
v_int_hidden = results_neur["neur_hidden0_int_pop_v"]

v_eff_int_hidden = v_int_hidden * gd_int / (gd_int + glk_int + gsom_int)

r_input = results_neur["neur_input_input_pop_r"]

vb_output = results_neur["neur_output_output_pop_vb"]
vnudge_output = results_neur["neur_output_output_pop_vnudge"]
u_output = results_neur["neur_output_output_pop_u"]
r_output = results_neur["neur_output_output_pop_r"]

vb_eff_output = vb_output * gb_out / (glk_out + gb_out + ga_out)

W_hp_ip = results_syn["syn_input_input_pop_to_hidden0_pyr_pop_g"][:, 0, 0]
W_hi_hp = results_syn["syn_hidden0_pyr_pop_to_int_pop_g"][:, 0, 0]
W_hp_hi = results_syn["syn_hidden0_int_pop_to_pyr_pop_g"][:, 0, 0]
W_op_hp = results_syn["syn_hidden0_pyr_pop_to_output_output_pop_g"][:, 0, 0]
W_hp_op = results_syn["syn_output_output_pop_to_hidden0_pyr_pop_g"][:, 0, 0]

W_po = results_syn["syn_output_output_pop_to_hidden0_pyr_pop_g"]
W_pi = results_syn["syn_hidden0_int_pop_to_pyr_pop_g"]

fig, ax = plt.subplots(1, 1)

ax.plot(T_AX_READOUT, W_hp_ip, label="hp_ip")
ax.plot(T_AX_READOUT, W_hi_hp, label="hi_hp")
ax.plot(T_AX_READOUT, W_hp_hi, label="hp_hi")
ax.plot(T_AX_READOUT, W_op_hp, label="op_hp")
ax.plot(T_AX_READOUT, W_hp_op, label="hp_op")

ax.legend()

plt.show()

loss = ((vb_eff_output-vnudge_output)**2.).mean(axis=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(T_AX_READOUT[::T_SKIP_PLOT], loss[::T_SKIP_PLOT])
ax[0].set_xlabel("Training Steps")
ax[0].set_ylabel("Mean Squ. Err.")

ax[1].plot(T_AX_READOUT[::T_SKIP_PLOT], loss[::T_SKIP_PLOT])
ax[1].set_xlabel("Training Steps")
ax[1].set_ylabel("Mean Squ. Err.")

ax[1].set_yscale("log")

fig.tight_layout()

fig.savefig("./tests/teacher_network/plots/train_loss.png", dpi=600)


for k in range(T_SIGN_VALIDATION.shape[0]):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.step(t_ax_val, np.reshape(targ_output_validation[k],(N_VAL_PATTERNS,N_OUT))
            [:, 0], where="post", label=r'$u_{trg}$')
    ax.plot(T_AX_READOUT_VAL,
            results_validation[k]["neur_output_output_pop_u"][:, 0],
            label=r'$\hat{v}_b$')

    ax.set_title(f'Validation at Time Step $t = {T_SIGN_VALIDATION[k]}$')

    ax.set_xlabel(r"$Time Steps$")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        f'./tests/teacher_network/plots/validation_t_{T_SIGN_VALIDATION[k]}.png',
        dpi=300)

    plt.show()

pdb.set_trace()
