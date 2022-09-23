#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

from mc.defaults import default_network_params

from mc.network import Network

from pygenn.genn_model import init_var

import pdb

dim_in = 30
dim_hidden = 20
dim_out = 10

glk_exc = default_network_params["exc_model_init"]["parameters"]["glk"]
gb_exc = default_network_params["exc_model_init"]["parameters"]["gb"]
ga_exc = default_network_params["exc_model_init"]["parameters"]["ga"]

vb_att_fact_exc = gb_exc / (glk_exc + gb_exc + ga_exc)

glk_int = default_network_params["int_model_init"]["parameters"]["glk"]
gd_int = default_network_params["int_model_init"]["parameters"]["gd"]

v_att_fact_int = gd_int / (glk_int + gd_int)

glk_out = default_network_params["output_model_init"]["parameters"]["glk"]
gb_out = default_network_params["output_model_init"]["parameters"]["gb"]
ga_out = default_network_params["output_model_init"]["parameters"]["ga"]
gsom_out = default_network_params["output_model_init"]["parameters"]["gsom"]

vb_att_fact_out = gb_out / (glk_out + gb_out + ga_out)

def phi(x):
    #return x
    return np.log(1.+np.exp(1.*(x)))

Net = Network({"dim_input": dim_in,
                "dim_hidden": [dim_hidden],
                "dim_output": dim_out,
                "synapse_output_to_hidden_exc_init": {
                    "variables": {
                        "g": init_var("Uniform", 
                            {"min": -1.0/np.sqrt(dim_out), 
                            "max": 1.0/np.sqrt(dim_out)})
                    },
                    "parameters": {
                        "muPP": 6e-3
                    }

                },
                "synapse_hidden_exc_to_output_init": {
                    "variables": {
                        "g": init_var("Uniform", 
                            {"min": -1.0/np.sqrt(dim_hidden), 
                            "max": 1.0/np.sqrt(dim_hidden)}),
                        "vbEff": 0.0
                    },
                    "parameters": {
                        "muPP": 6e-3
                    }

                }})

N_SAMPLES = 3500
TIME_STEPS_PRESENT_SAMPLE = 200
TIME_STEPS = N_SAMPLES * TIME_STEPS_PRESENT_SAMPLE

W_10 = (np.random.rand(dim_in,dim_hidden)-0.5)*2./np.sqrt(dim_in)
W_21 = (np.random.rand(dim_hidden,dim_out)-0.5)*2./np.sqrt(dim_hidden)

input_test = 2.*(np.random.rand(N_SAMPLES,dim_in)-0.5).repeat(TIME_STEPS_PRESENT_SAMPLE,axis=0)

output_trg_test = phi(input_test @ W_10) @ W_21

#input_data = np.zeros((TIME_STEPS,Net.dim_input_layer))

neur_results, weights_results = Net.run_network_top_down_input(input_test,output_trg_test,
                                    [("input_pop","r"),("hidden_exc_pop0","u"),("hidden_int_pop0","u"),
                                    ("hidden_exc_pop0","va"),("hidden_exc_pop0","vb"),
                                    ("output_pop","vb"),("output_pop","u"),
                                    ("hidden_exc_pop0","r"),("hidden_int_pop0","r"),("hidden_int_pop0","v"),
                                    ("output_pop","r")],
                                    [("syn_hidden_exc_to_output","g"),
                                    ("syn_hidden_int_to_hidden_exc0","g"),
                                    ("syn_hidden_exc_to_hidden_int0","g"),
                                    ("syn_input_to_hidden_exc","g")])#,
'''
[("syn_input_to_hidden_exc","g"),
("syn_hidden_exc_to_hidden_int0","g"),
("syn_hidden_int_to_hidden_exc0","g"),
("syn_hidden_exc_to_output","g"),
("syn_output_to_hidden_exc","g")])'''

vb_out_hat = neur_results["output_pop_vb"] * vb_att_fact_out

plt.plot(((vb_out_hat-output_trg_test)**2.)[TIME_STEPS_PRESENT_SAMPLE-2::TIME_STEPS_PRESENT_SAMPLE].mean(axis=1))

import pdb
pdb.set_trace()

for k in range(dim_out):
    plt.plot(output_trg_test[:,k],'-',linewidth=0.5,c=color_cycle[k])
    plt.plot(neur_results["output_pop_vb"][:,k],'--',linewidth=0.5,c=color_cycle[k])
#plt.plot(neur_results["output_pop_vb"],c="tab:green")

import pdb
pdb.set_trace()



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
