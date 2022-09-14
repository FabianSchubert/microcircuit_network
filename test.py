#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

from mc.network import Network

import pdb

Net = Network({})

TIME_STEPS = 1000

t_ax = np.linspace(0.,50.,TIME_STEPS)

input_data = np.sin(t_ax)[:,np.newaxis].repeat(Net.dim_input_layer,axis=1)
print(input_data.shape)
#input_data = np.zeros((TIME_STEPS,Net.dim_input_layer))

r_in_rec, r_h_rec, r_out_rec = Net.run_network_static_weights(input_data)

for k in range(Net.dim_input_layer):
    plt.plot(r_in_rec[:,k],color=color_cycle[0],linewidth=1,label=('input layer' if k==0 else None))

for k in range(Net.dim_hidden_layers[0]):
    plt.plot(r_h_rec[:,k],color=color_cycle[1],linewidth=1,label=('hidden layer' if k==0 else None))

for k in range(Net.dim_output_layer):
    plt.plot(r_out_rec[:,k],color=color_cycle[2],linewidth=1,label=('output layer' if k==0 else None))

plt.legend()

pdb.set_trace()
