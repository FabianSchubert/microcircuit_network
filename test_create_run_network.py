#! /usr/bin/env python3

from mc.rate_mc.network import Network

from pygenn.genn_model import GeNNModel

import ipdb
import pdb

import numpy as np

import matplotlib.pyplot as plt

def phi(x):
    #return x
    return np.log(1.+np.exp(x))
    #return np.tanh(x)

n_in = 4
n_hidden = [3]
n_out = 2

n_hidden_teacher = 3

T_show_patterns = 250
n_patterns = 2000
T_skip = 10
T = T_show_patterns * n_patterns
T_rec = int(T/T_skip)

T_init_zero = T_show_patterns * 500

t_ax = np.arange(T)
t_ax_skip = t_ax[::T_skip]

test_input = np.random.rand(n_patterns,n_in)
test_input = np.repeat(test_input,T_show_patterns,axis=0)
test_input[:T_init_zero,:] = 0.

W_10 = (np.random.rand(n_hidden_teacher,n_in)-0.5)/np.sqrt(n_in)
W_21 = (np.random.rand(n_out,n_hidden_teacher)-0.5)/np.sqrt(n_hidden_teacher)

test_output = (W_21 @ phi(W_10 @ test_input.T)).T

test_input_tuple = (test_input,"neur_input_input_pop","u")
test_output_tuple = (test_output,"neur_output_pyr_pop","vnudge")

net = Network("testnet",n_in,n_hidden,n_out,dt=0.1)

# Manually set the feedback weight to be the transpose of the
# teacher feed-forward...
# NOTE: In genn, weights are stored as flattened arrays with
# c-order (last index changes first), but the underlying 2d-matrix
# is of dimension <size source> x <size target>...so you just need to
# push W_21.flatten() to the device rather than W_21.T.flatten. 

synview = net.syn_pops["syn_output_pyr_pop_to_hidden0_pyr_pop"].vars["g"].view
synview[:] = W_21.flatten()
net.syn_pops["syn_output_pyr_pop_to_hidden0_pyr_pop"].push_var_to_device("g")

ipdb.set_trace()

results_neur, results_syn = net.run_sim([test_input_tuple,test_output_tuple],
    [("neur_hidden0_pyr_pop","u"),
    ("neur_output_pyr_pop","vb"),
    ("neur_output_pyr_pop","vnudge"),
    ("neur_output_pyr_pop","u"),
    ("neur_hidden0_pyr_pop","va_exc"),
    ("neur_hidden0_pyr_pop","va_int"),
    ("neur_hidden0_pyr_pop","vb"),
    ("neur_hidden0_pyr_pop","va"),
    ("neur_hidden0_int_pop","r"),
    ("neur_hidden0_int_pop","u"),
    ("neur_hidden0_int_pop","v"),
    ("neur_hidden0_pyr_pop","r"),
    ("neur_output_pyr_pop","r"),
    ("neur_input_input_pop","r")],
    [("syn_hidden0_int_pop_to_pyr_pop","g"),
    ("syn_hidden0_pyr_pop_to_int_pop","g"),
    ("syn_hidden0_pyr_pop_to_output_pyr_pop","g"),
    ("syn_output_pyr_pop_to_hidden0_pyr_pop","g"),
    ("syn_input_input_pop_to_hidden0_pyr_pop","g")],
    T_skip=T_skip)

#ipdb.set_trace()

va_exc_hidden = results_neur["neur_hidden0_pyr_pop_va_exc"]
va_int_hidden = results_neur["neur_hidden0_pyr_pop_va_int"]
va_hidden = results_neur["neur_hidden0_pyr_pop_va"]
vb_hidden = results_neur["neur_hidden0_pyr_pop_vb"]
r_int_hidden = results_neur["neur_hidden0_int_pop_r"]
r_pyr_hidden = results_neur["neur_hidden0_pyr_pop_r"]

u_int_hidden = results_neur["neur_hidden0_int_pop_u"]
v_int_hidden = results_neur["neur_hidden0_int_pop_v"]

v_eff_int_hidden = v_int_hidden * 1.0/(0.1 + 1.0)

r_input = results_neur["neur_input_input_pop_r"]

vb_output = results_neur["neur_output_pyr_pop_vb"]
vnudge_output = results_neur["neur_output_pyr_pop_vnudge"]
u_output = results_neur["neur_output_pyr_pop_u"]
r_output = results_neur["neur_output_pyr_pop_r"]

vb_eff_output = vb_output * 1.0 / (0.1 + 1.0)

W_hp_ip = results_syn["syn_input_input_pop_to_hidden0_pyr_pop_g"][:,0,0]
W_hi_hp = results_syn["syn_hidden0_pyr_pop_to_int_pop_g"][:,0,0]
W_hp_hi = results_syn["syn_hidden0_int_pop_to_pyr_pop_g"][:,0,0]
W_op_hp = results_syn["syn_hidden0_pyr_pop_to_output_pyr_pop_g"][:,0,0]
W_hp_op = results_syn["syn_output_pyr_pop_to_hidden0_pyr_pop_g"][:,0,0]

fig, ax = plt.subplots(1,1)

ax.plot(W_hp_ip,label="hp_ip")
ax.plot(W_hi_hp,label="hi_hp")
ax.plot(W_hp_hi,label="hp_hi")
ax.plot(W_op_hp,label="op_hp")
ax.plot(W_hp_op,label="hp_op")

ax.legend()

plt.show()


loss = ((vb_eff_output-test_output[::T_skip])**2.).mean(axis=1)

T_show_patterns_rec = int(T_show_patterns/T_skip)

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot(t_ax_skip,loss)
ax[0].set_xlabel("Training Steps")
ax[0].set_ylabel("Mean Squ. Err.")

ax[1].plot(t_ax_skip,loss)
ax[1].set_xlabel("Training Steps")
ax[1].set_ylabel("Mean Squ. Err.")

ax[1].set_yscale("log")

fig.tight_layout()

fig.savefig("train_loss.png",dpi=600)

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot(t_ax_skip,vb_eff_output[:,0],label=r'$\hat{v}_b$')
ax[0].plot(t_ax_skip,test_output[::T_skip,0],label=r'$u_{trg}$')

ax[0].legend()

ax[0].set_xlabel("Training Steps")

ax[1].plot(t_ax_skip,vb_eff_output[:,0],label=r'$\hat{v}_b$')
ax[1].plot(t_ax_skip,test_output[::T_skip,0],label=r'$u_{trg}$')

ax[1].legend()

ax[1].set_xlim([T-10000,T])

ax[1].set_xlabel("Training Steps")

fig.tight_layout()

fig.savefig("u_trg_vs_vb_eff.png",dpi=600)

plt.show()


pdb.set_trace()
