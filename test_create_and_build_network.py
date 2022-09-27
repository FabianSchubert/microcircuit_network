#! /usr/bin/env python3

from mc.network import NetworkLayer as NL
from mc.network import Network

from pygenn.genn_model import GeNNModel

from mc.defaults import default_neuron, default_synapse

model = GeNNModel("float","testmodel")

network_def = {
    "name": "testnet",

    "layers": [
        {
            "layer_name": "testlayer1",
            
            "neur_pops":[
                ({"pop_name": "pop1",
                "num_neurons": 10} | default_neuron)
            ],
            
            "syn_pops": [
                ({"source": "pop1",
                "target": "pop1",
                } | default_synapse)
            ]
        },
        {
            "layer_name": "testlayer2",
            
            "neur_pops":[
                ({"pop_name": "pop1",
                "num_neurons": 10} | default_neuron)
            ],
            
            "syn_pops": [
                ({"source": "pop1",
                "target": "pop1",
                } | default_synapse)
            ]
        }
    ],

    "cross_layer_syn_pops": [
        ({
            "source_layer": "testlayer1",
            
            "source_pop": "pop1",
            
            "target_layer": "testlayer2",

            "target_pop": "pop1"
        } | default_synapse)
    ]
}

net = Network(network_def)

import pdb
pdb.set_trace()
