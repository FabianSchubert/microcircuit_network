#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

from mc.network import Network

import pdb



Net = Network({})

N_SAMPLES = 500
TIME_STEPS_PRESENT_SAMPLE = 200
TIME_STEPS = N_SAMPLES * TIME_STEPS_PRESENT_SAMPLE

test_lin_model = np.random.rand(2,1)

input_test = np.random.normal(0.,1.,(N_SAMPLES,2)).repeat(TIME_STEPS_PRESENT_SAMPLE,axis=0)
output_trg_test = input_test @ test_lin_model

t_ax = np.linspace(0.,50.,TIME_STEPS)

input_data = np.sin(t_ax)[:,np.newaxis].repeat(Net.dim_input_layer,axis=1)
print(input_data.shape)
#input_data = np.zeros((TIME_STEPS,Net.dim_input_layer))

r_in_rec, r_h_rec, vb_out_rec, w_h_to_out_rec, w_out_to_h_rec, w_in_to_h_rec = Net.run_network_top_down_input(input_test,output_trg_test)

fig, ax = plt.subplots(1,2)

for k in range(Net.dim_input_layer):
    ax[0].plot(r_in_rec[:,k],color=color_cycle[0],linewidth=1,label=('input layer' if k==0 else None))

for k in range(Net.dim_hidden_layers[0]):
    ax[0].plot(r_h_rec[:,k],color=color_cycle[1],linewidth=1,label=('hidden layer' if k==0 else None))

for k in range(Net.dim_output_layer):
    ax[0].plot(vb_out_rec[:,k],color=color_cycle[2],linewidth=1,label=('output layer' if k==0 else None))

ax[0].legend()

ax[1].plot(vb_out_rec[:,0])
ax[1].plot(output_trg_test)

fig_w, ax_w = plt.subplots()

ax_w.plot(w_rec[:,0,:])

pdb.set_trace()
