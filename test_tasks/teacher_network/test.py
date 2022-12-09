'''
This test trains a microcircuit model
with on hidden layer on pairs of input
and target data that was generated using
a randomly teacher network with one hidden
layer and randomly generated weights.
'''

import cProfile
import io
import pdb
import pstats
from pstats import SortKey

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mc.mc.network import Network
from mc.mc.neurons.params import (int_param_space,
                                  output_param_space,
                                  pyr_hidden_param_space)

from .utils import gen_input_output_data

from test_tasks.utils import calc_loss_interp

col_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

GB_PYR = pyr_hidden_param_space["gb"]
GA_PYR = pyr_hidden_param_space["ga"]
GLK_PYR = pyr_hidden_param_space["glk"]

GB_OUT = output_param_space["gb"]
GA_OUT = output_param_space["ga"]
GLK_OUT = output_param_space["glk"]

GLK_INT = int_param_space["glk"]
GD_INT = int_param_space["gd"]
GSOM_INT = int_param_space["gsom"]

##################
# Model parameters
DT = 0.5
##################

#####################
# Training parameters
N_IN = 30
N_HIDDEN = [20]
N_OUT = 10

N_HIDDEN_TEACHER = 20

T_SHOW_PATTERNS = 100
N_PATTERNS = 200000

T_OFFSET = 0

T = N_PATTERNS * T_SHOW_PATTERNS + T_OFFSET
####################

######################
# recording parameters

T_SKIP_REC = T_SHOW_PATTERNS * 50
T_OFFSET_REC = 140
T_AX_READOUT = np.arange(T)[::T_SKIP_REC] + T_OFFSET_REC

T_SPIKE_REC = T # size of the spike recording buffer
######################

#####################
# plotting parameters

# skip readout data
T_SKIP_PLOT = 10
#####################

#######################
# validation parameters

N_VALIDATION = 20

T_INTERVAL_VALIDATION = int(T / N_VALIDATION)

N_VAL_PATTERNS = 200

T_RUN_VALIDATION = T_SHOW_PATTERNS * N_VAL_PATTERNS

T_SIGN_VALIDATION = np.arange(T)[::T_INTERVAL_VALIDATION]
N_VALIDATION = T_SIGN_VALIDATION.shape[0]

T_SKIP_REC_VAL = 2
T_OFFSET_REC_VAL = 0
T_AX_READOUT_VAL = np.arange(T_RUN_VALIDATION)[::T_SKIP_REC_VAL] \
                   + T_OFFSET_REC_VAL

T_SPIKE_REC_VALIDATION = T_RUN_VALIDATION # size of the validation spike recording buffer
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

# generate a single validataion example



for k in range(N_VALIDATION):


    (t_ax_val, val_input,
     val_output, _, _) = gen_input_output_data(N_IN,
                                               N_HIDDEN_TEACHER,
                                               N_OUT,
                                               N_VAL_PATTERNS,
                                               T_SHOW_PATTERNS,
                                               0,
                                               (W_10, W_21))

    # {T, t_sign, ext_data_input, ext_data_pop_vars, readout_neur_pop_vars}

    _dict_data_validation = {}

    _dict_data_validation["T"] = T_RUN_VALIDATION
    _dict_data_validation["t_sign"] = t_ax_val
    _dict_data_validation["ext_data_input"] = val_input
    _dict_data_validation["ext_data_pop_vars"] = [
        (np.zeros((1, N_OUT)),
         np.array([0.]).astype("int"),
         "neur_output_output_pop", "gnudge")]
    # This sets the nudging conductance of the output to zero
    # at the beginning of the evaluation.

    _dict_data_validation["readout_neur_pop_vars"] = [("neur_output_output_pop",
                                                       "u", T_AX_READOUT_VAL),
                                                      ("neur_input_input_pop",
                                                       "r", T_AX_READOUT_VAL),
                                                      ("neur_hidden0_pyr_pop",
                                                       "vb", T_AX_READOUT_VAL)]

    data_validation.append(_dict_data_validation)

    targ_output_validation.append(val_output)

#######################################
# set populations to record spikes from
neur_pops_spike_rec = ["neur_output_output_pop"]
neur_pops_spike_rec_val = ["neur_output_output_pop"]
#######################################

##########################
# Initialize network

net = Network("testnet", N_IN, N_HIDDEN, N_OUT,
              N_PATTERNS,
              T_SPIKE_REC,
              T_SPIKE_REC_VALIDATION,
              dt=DT,
              spike_rec_pops=neur_pops_spike_rec,
              spike_rec_pops_val=neur_pops_spike_rec_val,
              plastic=True, t_inp_static_max=N_VAL_PATTERNS)

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

synview_ip = synview_pp_fwd * (GB_PYR + GLK_PYR) / (GB_PYR + GA_PYR + GLK_PYR)

net.syn_pops["syn_hidden0_pyr_pop_to_int_pop"].push_var_to_device("g")

#################################

##################################
# prepare the readout instructions
neur_readout_list = [#("neur_output_output_pop",
                     # "vb", T_AX_READOUT),
                     #("neur_output_output_pop",
                     # "vnudge", T_AX_READOUT),
                     #("neur_output_output_pop",
                     # "u", T_AX_READOUT),
                     # ("neur_output_output_pop",
                     # "r", T_AX_READOUT),
                     #("neur_hidden0_pyr_pop",
                     # "u", T_AX_READOUT),
                     # ("neur_hidden0_pyr_pop",
                     # "va_exc", T_AX_READOUT),
                     # ("neur_hidden0_pyr_pop",
                     # "va_int", T_AX_READOUT),
                     # ("neur_hidden0_pyr_pop",
                     # "vb", T_AX_READOUT),
                     # ("neur_hidden0_pyr_pop",
                     # "va", T_AX_READOUT),
                     # ("neur_hidden0_int_pop",
                     # "r", T_AX_READOUT),
                     # ("neur_hidden0_int_pop",
                     # "u", T_AX_READOUT),
                     # ("neur_hidden0_int_pop",
                     # "v", T_AX_READOUT),
                     #("neur_hidden0_pyr_pop",
                     # "r", T_AX_READOUT),
                     #("neur_input_input_pop",
                     # "r", T_AX_READOUT)
                     ]

syn_readout_list = []
'''[("syn_hidden0_int_pop_to_pyr_pop",
 "g", T_AX_READOUT),
("syn_hidden0_pyr_pop_to_int_pop",
 "g", T_AX_READOUT),
("syn_hidden0_pyr_pop_to_output_output_pop",
 "g", T_AX_READOUT),
("syn_output_output_pop_to_hidden0_pyr_pop",
 "g", T_AX_READOUT),
("syn_input_input_pop_to_hidden0_pyr_pop",
 "g", T_AX_READOUT)]'''

pr = cProfile.Profile()
pr.enable()

(results_neur, results_syn,
 results_spikes,
 results_validation) = net.run_sim(T,
                                   t_ax_train,
                                   test_input, test_output,
                                   ext_data_pop_vars,
                                   neur_readout_list,
                                   syn_readout_list,
                                   T_SIGN_VALIDATION,
                                   data_validation,
                                   show_progress_val=False)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())


'''
va_exc_hidden = results_neur["neur_hidden0_pyr_pop_va_exc"]
va_int_hidden = results_neur["neur_hidden0_pyr_pop_va_int"]
va_hidden = results_neur["neur_hidden0_pyr_pop_va"]
vb_hidden = results_neur["neur_hidden0_pyr_pop_vb"]
r_int_hidden = results_neur["neur_hidden0_int_pop_r"]

u_int_hidden = results_neur["neur_hidden0_int_pop_u"]
v_int_hidden = results_neur["neur_hidden0_int_pop_v"]

v_eff_int_hidden = v_int_hidden * GD_INT / (GD_INT + GLK_INT + GSOM_INT)

'''

# u_hidden = results_neur["neur_hidden0_pyr_pop_u"]
# r_input = results_neur["neur_input_input_pop_r"]

# r_pyr_hidden = results_neur["neur_hidden0_pyr_pop_r"]

#vb_output = results_neur["neur_output_output_pop_vb"]
#vnudge_output = results_neur["neur_output_output_pop_vnudge"]
#u_output = results_neur["neur_output_output_pop_u"]
# r_output = results_neur["neur_output_output_pop_r"]

#vb_eff_output = vb_output * GB_OUT / (GLK_OUT + GB_OUT + GA_OUT)

'''
# plot weights
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
'''


loss = np.ndarray((T_SIGN_VALIDATION.shape[0]))

for k in range(T_SIGN_VALIDATION.shape[0]):
    _dat_targ = np.reshape(targ_output_validation[k], (N_VAL_PATTERNS, N_OUT))
    _dat_readout = results_validation[k]["neur_var_rec"]["neur_output_output_pop_u"]

    _loss = calc_loss_interp(t_ax_val, T_AX_READOUT_VAL,
                             _dat_targ, _dat_readout)
    loss[k] = _loss

# loss = ((vb_eff_output - vnudge_output) ** 2.).mean(axis=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(T_SIGN_VALIDATION, loss)
ax[0].set_xlabel("Training Steps")
ax[0].set_ylabel("Mean Squ. Err.")

ax[1].plot(T_SIGN_VALIDATION, loss)
ax[1].set_xlabel("Training Steps")
ax[1].set_ylabel("Mean Squ. Err.")

ax[1].set_yscale("log")

fig.tight_layout()

fig.savefig("./test_tasks/teacher_network/plots/train_loss.png", dpi=600)

plt.show()


for k in range(T_SIGN_VALIDATION.shape[0]):
    spike_times, spike_indices = results_validation[k]["spike_rec"]["neur_output_output_pop"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    stpplot = ax.step(t_ax_val*DT, np.reshape(targ_output_validation[k], (N_VAL_PATTERNS, N_OUT))[:, 0],
                      where="post", label=r'$u_{trg}$', zorder=5)

    lnplot = ax.plot(T_AX_READOUT_VAL*DT,
                     results_validation[k]["neur_var_rec"]["neur_output_output_pop_u"][:, 0],
                     label=r'$\hat{v}_b$', zorder=10)

    ax.vlines(x=spike_times[np.where(spike_indices == 0)[0]],
              colors="k",
              ymin=ax.get_ylim()[0],
              ymax=ax.get_ylim()[1],
              zorder=0, alpha=0.25,
              linewidth=0.5)

    ax.set_title(f'Validation at Time $t = {T_SIGN_VALIDATION[k]*DT}$')

    ax.set_xlabel(r"$Time$")

    ax.set_xlim([0.,500.])

    ax.legend()

    fig.tight_layout()
    
    fig.savefig(
        f'./test_tasks/teacher_network/plots/validation_t_{T_SIGN_VALIDATION[k]}.png',
        dpi=300)
    
    plt.close()
    #plt.show()

T_HIST_SPIKES = 600.
N_BINS_SPIKES = 200

spike_times = results_spikes["neur_output_output_pop"][0]

spike_times_val = results_validation[int(N_VALIDATION/2)]["spike_rec"]["neur_output_output_pop"][0]

# find the time where new input is presented that
# is closest to spike_times.max() * 0.5,
# so that the histogram starts at the onset of
# new input.
T_START_BIN = spike_times.max() * 0.5
T_START_BIN = t_ax_train[np.abs(T_START_BIN - t_ax_train*DT).argmin()]*DT

T_START_BIN_VAL = 0.0

spike_bins = np.linspace(T_START_BIN, T_START_BIN + T_HIST_SPIKES, N_BINS_SPIKES + 1)
spike_bins_val = np.linspace(T_START_BIN_VAL, T_START_BIN_VAL + T_HIST_SPIKES, N_BINS_SPIKES + 1)

h_spikes = np.histogram(spike_times, bins=spike_bins)
rate_est_spikes = h_spikes[0] * N_BINS_SPIKES / (T_HIST_SPIKES * N_OUT)

h_spikes_val = np.histogram(spike_times_val, bins=spike_bins_val)
rate_est_spikes_val = h_spikes_val[0] * N_BINS_SPIKES / (T_HIST_SPIKES * N_OUT)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.stairs(rate_est_spikes, h_spikes[1] - T_START_BIN,
           fill=True, facecolor=col_cycle[0]+"50",
           edgecolor=col_cycle[0], linewidth=1.5, label="plast")

ax.stairs(rate_est_spikes_val, h_spikes_val[1] - T_START_BIN_VAL,
           fill=True, facecolor=col_cycle[1]+"50",
           edgecolor=col_cycle[1], linewidth=1.5, label="val")

ax.set_ylabel("avg. spike rate")
ax.set_xlabel("time")

ax.legend()

fig.tight_layout()

fig.savefig("./test_tasks/teacher_network/plots/spike_rates.png", dpi=300)

plt.show()

# W_op_hp = results_syn["syn_hidden0_pyr_pop_to_output_output_pop_g"]
# I_vb = np.ndarray(vb_output.shape)

# for t in tqdm(range(I_vb.shape[0])):
#     I_vb[t] = W_op_hp[t] @ r_pyr_hidden[t]

plt.ion()
pdb.set_trace()
