#! /usr/bin/env python3

import ipdb
import matplotlib.pyplot as plt
import numpy as np

import pdb

from mc.rate_mc.network import Network

from mc.rate_mc.neurons.params import (pyr_hidden_param_space,
                                       output_param_space,
                                       int_param_space)

from utils import gen_input_output_data

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
N_PATTERNS = 100000

T_OFFSET = 0

T = N_PATTERNS * T_SHOW_PATTERNS + T_OFFSET
####################

######################
# recording parameters

T_SKIP_REC = 450
T_OFFSET_REC = 140
T_AX_READOUT = np.arange(T)[::T_SKIP_REC]+T_OFFSET_REC
######################

#####################
# plotting parameters

# skip readout data
T_SKIP_PLOT = 30
#####################

#######################
# validation parameters

T_INTERVAL_VALIDATION = int(T/10)

N_VAL_PATTERNS = 20

T_RUN_VALIDATION = T_SHOW_PATTERNS * N_VAL_PATTERNS

T_SIGN_VALIDATION = np.arange(T)[::T_INTERVAL_VALIDATION]
N_VALIDATION = T_SIGN_VALIDATION.shape[0]
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

########################

##########################
# generate validation data
data_validation = []

targ_output_validation = []

for k in range(n_validation):

    (t_ax_val, val_input,
     val_output, _, _) = gen_input_output_data(N_IN,
                                               N_HIDDEN_TEACHER,
                                               N_OUT,
                                               N_VAL_PATTERNS,
                                               T_SHOW_PATTERNS,
                                               0,
                                               (W_10, W_21))

    {T, t_sign, ext_data_input, readout_neur_pop_vars}

    _dict_data_validation = {}

    _dict_data_validation["T"] = T_RUN_VALIDATION
    _dict_data_validation["t_sign"] = t_ax_val
    _dict_data_validation["ext_data_input"] = val_input
    _dict_data_validation[]

    # the second input tuple sets the nudging
    # conductance in the output population to zero
    # at the start of the validation.
    _val_input_tuples = [(val_input, t_ax_val, "neur_input_input_pop", "u"),
                         (np.zeros((1, n_out)), np.array([0.]).astype("int"),
                          "neur_output_output_pop", "gnudge")]


for k in range(t_validation.shape[0]):
    val_input = np.random.rand(n_val_patterns, n_in)

    val_target = (W_21 @ phi(W_10 @ val_input.T)).T

    target_data_validation.append(val_target)

    t_ax_val = np.arange(n_val_patterns)*T_show_patterns

    t_ax_readout_val = np.arange(T_run_validation)

    # the second input tuple sets the nudging
    # conductance in the output population to zero
    # at the start of the validation.
    val_input_tuple = [(val_input, t_ax_val, "neur_input_input_pop", "u"),
                       (np.zeros((1, n_out)), np.array([0.]).astype("int"),
                       "neur_output_output_pop", "gnudge")]

    ext_data_validation.append(val_input_tuple)

    readout_neur_pop_vars_validation.append(
        [("neur_output_output_pop", "u", t_ax_readout_val)])

    readout_syn_pop_vars_validation.append(
        [("syn_hidden0_pyr_pop_to_output_output_pop", "g", t_ax_readout_val)])

data_validation = [t_validation, T_run_validation_list,
                   ext_data_validation, readout_neur_pop_vars_validation,
                   readout_syn_pop_vars_validation]
###############

net = Network("testnet", n_in, n_hidden, n_out, dt=0.5)

ipdb.set_trace()

# Manually initialize the network in the self-predicting state

# W_pi = - W_pp_back
synview_pi = net.syn_pops["syn_hidden0_int_pop_to_pyr_pop"].vars["g"].view
synview_pp_back = net.syn_pops["syn_output_output_pop_to_hidden0_pyr_pop"
                               ].vars["g"].view
net.syn_pops["syn_output_output_pop_to_hidden0_pyr_pop"
             ].pull_var_from_device("g")
synview_pi[:] = -synview_pp_back
net.syn_pops["syn_hidden0_int_pop_to_pyr_pop"].push_var_to_device("g")

# W_ip = (gb+glk)/(gb + ga + glk) * W_pp_fwd
synview_ip = net.syn_pops["syn_hidden0_pyr_pop_to_int_pop"].vars["g"].view
synview_pp_fwd = net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"
                              ].vars["g"].view
net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"
             ].pull_var_from_device("g")
synview_ip = synview_pp_fwd * (gb_pyr + glk_pyr) / (gb_pyr + ga_pyr + glk_pyr)
net.syn_pops["syn_hidden0_pyr_pop_to_int_pop"].push_var_to_device("g")

input_list = [test_input_tuple,
              test_output_tuple,
              switch_plast_input_tuple]

neur_readout_list = [("neur_output_output_pop", "vb", t_ax_readout),
                     ("neur_output_output_pop",
                     "vnudge", t_ax_readout),
                     ("neur_output_output_pop",
                      "u", t_ax_readout),
                     ("neur_output_output_pop",
                      "r", t_ax_readout),
                     ("neur_hidden0_pyr_pop",
                      "u", t_ax_readout),
                     ("neur_hidden0_pyr_pop",
                      "va_exc", t_ax_readout),
                     ("neur_hidden0_pyr_pop",
                      "va_int", t_ax_readout),
                     ("neur_hidden0_pyr_pop",
                      "vb", t_ax_readout),
                     ("neur_hidden0_pyr_pop",
                      "va", t_ax_readout),
                     ("neur_hidden0_int_pop",
                      "r", t_ax_readout),
                     ("neur_hidden0_int_pop",
                      "u", t_ax_readout),
                     ("neur_hidden0_int_pop",
                      "v", t_ax_readout),
                     ("neur_hidden0_pyr_pop",
                      "r", t_ax_readout),
                     ("neur_input_input_pop", "r", t_ax_readout)]

syn_readout_list = [("syn_hidden0_int_pop_to_pyr_pop",
                    "g", t_ax_readout),
                    ("syn_hidden0_pyr_pop_to_int_pop",
                    "g", t_ax_readout),
                    ("syn_hidden0_pyr_pop_to_output_output_pop",
                    "g", t_ax_readout),
                    ("syn_output_output_pop_to_hidden0_pyr_pop",
                    "g", t_ax_readout),
                    ("syn_input_input_pop_to_hidden0_pyr_pop",
                    "g", t_ax_readout)]

(results_neur, results_syn,
 results_validation) = net.run_sim(T, input_list,
                                   neur_readout_list,
                                   syn_readout_list,
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

ax.plot(t_ax_readout, W_hp_ip, label="hp_ip")
ax.plot(t_ax_readout, W_hi_hp, label="hi_hp")
ax.plot(t_ax_readout, W_hp_hi, label="hp_hi")
ax.plot(t_ax_readout, W_op_hp, label="op_hp")
ax.plot(t_ax_readout, W_hp_op, label="hp_op")

ax.legend()

plt.show()

loss = ((vb_eff_output-vnudge_output)**2.).mean(axis=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(t_ax_readout[::T_skip_plot], loss[::T_skip_plot])
ax[0].set_xlabel("Training Steps")
ax[0].set_ylabel("Mean Squ. Err.")

ax[1].plot(t_ax_readout[::T_skip_plot], loss[::T_skip_plot])
ax[1].set_xlabel("Training Steps")
ax[1].set_ylabel("Mean Squ. Err.")

ax[1].set_yscale("log")

fig.tight_layout()

fig.savefig("./tests/teacher_network/plots/train_loss.png", dpi=600)

for k in range(t_validation.shape[0]):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.step(t_ax_val, target_data_validation[k]
            [:, 0], where="post", label=r'$u_{trg}$')
    ax.plot(t_ax_readout_val,
            results_validation[k][0]["neur_output_output_pop_u"][:, 0],
            label=r'$\hat{v}_b$')

    ax.set_title(f'Validation at Time Step $t = {t_validation[k]}$')

    ax.set_xlabel(r"$Time Steps$")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        f'./tests/teacher_network/plots/validation_t_{t_validation[k]}.png',
        dpi=300)

    plt.show()

pdb.set_trace()
