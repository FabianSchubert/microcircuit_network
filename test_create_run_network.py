#! /usr/bin/env python3

from mc.rate_mc.network import Network

from pygenn.genn_model import GeNNModel

from mc.rate_mc.neurons.params import (pyr_hidden_param_space,
                                    output_param_space,
                                    int_param_space,
                                    input_param_space)

gb_pyr = pyr_hidden_param_space["gb"]
ga_pyr = pyr_hidden_param_space["ga"]
glk_pyr = pyr_hidden_param_space["glk"]

gb_out = output_param_space["gb"]
ga_out = output_param_space["ga"]
glk_out = output_param_space["glk"]

glk_int = int_param_space["glk"]
gd_int = int_param_space["gd"]
gsom_int = int_param_space["gsom"]

import ipdb
import pdb

import numpy as np

import matplotlib.pyplot as plt

def phi(x):
    return x
    #return np.log(1.+np.exp(x))
    #return np.tanh(x)

n_in = 30
n_hidden = [50]
n_out = 10

n_hidden_teacher = 20

T_show_patterns = 150
n_patterns = 800000


T_switch_to_patterns = 1 * T_show_patterns

T = T_show_patterns * n_patterns

t_ax_input = np.arange(n_patterns)*T_show_patterns + T_switch_to_patterns

T_skip_rec = T_show_patterns*3
t_ax_readout = np.arange(T)[::T_skip_rec]+140

T_skip_plot = 30

test_input = np.random.rand(n_patterns,n_in)
#test_input_t_sign = np.arange(n_patterns)*T_show_patterns
#test_input = np.repeat(test_input,T_show_patterns,axis=0)[t_ax_skip]
#test_input[:T_init_zero,:] = 0.

W_10 = 4.*(np.random.rand(n_hidden_teacher,n_in)-0.5)/np.sqrt(n_in)
W_21 = 4.*(np.random.rand(n_out,n_hidden_teacher)-0.5)/np.sqrt(n_hidden_teacher)

test_output = (W_21 @ phi(W_10 @ test_input.T)).T

test_input_tuple = (test_input,t_ax_input,"neur_input_input_pop","u")
test_output_tuple = (test_output,t_ax_input,"neur_output_output_pop","vnudge")

switch_plast_input_dat = np.ones((2,n_out)) * 0.8
switch_plast_input_dat[0] = 0.

switch_plast_t = np.array([0,T_switch_to_patterns])

switch_plast_input_tuple = (switch_plast_input_dat,switch_plast_t,"neur_output_output_pop","gnudge")


net = Network("testnet",n_in,n_hidden,n_out,dt=0.1)

### Manually initialize the network in self-predicting state

# W_pi = - W_pp_back
synview_pi = net.syn_pops["syn_hidden0_int_pop_to_pyr_pop"].vars["g"].view
synview_pp_back = net.syn_pops["syn_output_output_pop_to_hidden0_pyr_pop"].vars["g"].view
net.syn_pops["syn_output_output_pop_to_hidden0_pyr_pop"].pull_var_from_device("g")
synview_pi[:] = -synview_pp_back
net.syn_pops["syn_hidden0_int_pop_to_pyr_pop"].push_var_to_device("g")

# W_ip = (gb+glk)/(gb + ga + glk) * W_pp_fwd
synview_ip = net.syn_pops["syn_hidden0_pyr_pop_to_int_pop"].vars["g"].view
synview_pp_fwd = net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"].vars["g"].view
net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"].pull_var_from_device("g")
synview_ip = synview_pp_fwd * (gb_pyr + glk_pyr) / (gb_pyr + ga_pyr + glk_pyr)
net.syn_pops["syn_hidden0_pyr_pop_to_int_pop"].push_var_to_device("g")

'''
# manually solve the problem for testing purposes...

synview = net.syn_pops["syn_input_input_pop_to_hidden0_pyr_pop"].vars["g"].view
synview[:] = W_10.T.flatten()*(0.1 + 1.0 + 0.8)/1.0
net.syn_pops["syn_input_input_pop_to_hidden0_pyr_pop"].push_var_to_device("g")

synview = net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"].vars["g"].view
synview[:] = W_21.T.flatten()*(0.1 + 1.0)/1.0
net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"].push_var_to_device("g")

synview = net.syn_pops["syn_hidden0_pyr_pop_to_int_pop"].vars["g"].view
synview[:] = W_21.T.flatten()*((0.1 + 1.0)/1.0)*(1.0+0.1)/(1.0+0.8+0.1)
'''

results_neur, results_syn = net.run_sim(T,[test_input_tuple,test_output_tuple,switch_plast_input_tuple],
    [("neur_output_output_pop","vb",t_ax_readout),
    ("neur_output_output_pop","vnudge",t_ax_readout),
    ("neur_output_output_pop","u",t_ax_readout),
    ("neur_output_output_pop","r",t_ax_readout),
    ("neur_hidden0_pyr_pop","u",t_ax_readout),
    ("neur_hidden0_pyr_pop","va_exc",t_ax_readout),
    ("neur_hidden0_pyr_pop","va_int",t_ax_readout),
    ("neur_hidden0_pyr_pop","vb",t_ax_readout),
    ("neur_hidden0_pyr_pop","va",t_ax_readout),
    ("neur_hidden0_int_pop","r",t_ax_readout),
    ("neur_hidden0_int_pop","u",t_ax_readout),
    ("neur_hidden0_int_pop","v",t_ax_readout),
    ("neur_hidden0_pyr_pop","r",t_ax_readout),
    ("neur_input_input_pop","r",t_ax_readout)],
    [("syn_hidden0_int_pop_to_pyr_pop","g",t_ax_readout),
    ("syn_hidden0_pyr_pop_to_int_pop","g",t_ax_readout),
    ("syn_hidden0_pyr_pop_to_output_output_pop","g",t_ax_readout),
    ("syn_output_output_pop_to_hidden0_pyr_pop","g",t_ax_readout),
    ("syn_input_input_pop_to_hidden0_pyr_pop","g",t_ax_readout)])

#ipdb.set_trace()

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

W_hp_ip = results_syn["syn_input_input_pop_to_hidden0_pyr_pop_g"][:,0,0]
W_hi_hp = results_syn["syn_hidden0_pyr_pop_to_int_pop_g"][:,0,0]
W_hp_hi = results_syn["syn_hidden0_int_pop_to_pyr_pop_g"][:,0,0]
W_op_hp = results_syn["syn_hidden0_pyr_pop_to_output_output_pop_g"][:,0,0]
W_hp_op = results_syn["syn_output_output_pop_to_hidden0_pyr_pop_g"][:,0,0]

W_po = results_syn["syn_output_output_pop_to_hidden0_pyr_pop_g"]
W_pi = results_syn["syn_hidden0_int_pop_to_pyr_pop_g"]

fig, ax = plt.subplots(1,1)

ax.plot(t_ax_readout,W_hp_ip,label="hp_ip")
ax.plot(t_ax_readout,W_hi_hp,label="hi_hp")
ax.plot(t_ax_readout,W_hp_hi,label="hp_hi")
ax.plot(t_ax_readout,W_op_hp,label="op_hp")
ax.plot(t_ax_readout,W_hp_op,label="hp_op")

ax.legend()

plt.show()

loss = ((vb_eff_output-vnudge_output)**2.).mean(axis=1)

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot(t_ax_readout[::T_skip_plot],loss[::T_skip_plot])
ax[0].set_xlabel("Training Steps")
ax[0].set_ylabel("Mean Squ. Err.")

ax[1].plot(t_ax_readout[::T_skip_plot],loss[::T_skip_plot])
ax[1].set_xlabel("Training Steps")
ax[1].set_ylabel("Mean Squ. Err.")

ax[1].set_yscale("log")

fig.tight_layout()

fig.savefig("train_loss.png",dpi=600)

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot(t_ax_readout[::T_skip_plot],vb_eff_output[:,0][::T_skip_plot],label=r'$\hat{v}_b$')
ax[0].plot(t_ax_readout[::T_skip_plot],vnudge_output[:,0][::T_skip_plot],label=r'$u_{trg}$')

ax[0].legend()

ax[0].set_xlabel("Training Steps")

ax[1].plot(t_ax_readout[::T_skip_plot],vb_eff_output[:,0][::T_skip_plot],label=r'$\hat{v}_b$')
ax[1].plot(t_ax_readout[::T_skip_plot],vnudge_output[:,0][::T_skip_plot],label=r'$u_{trg}$')

ax[1].legend()

ax[1].set_xlim([T-10000,T])

ax[1].set_xlabel("Training Steps")

fig.tight_layout()

fig.savefig("u_trg_vs_vb_eff.png",dpi=600)

plt.show()


pdb.set_trace()
