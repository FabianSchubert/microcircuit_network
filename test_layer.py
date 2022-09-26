#! /usr/bin/env python3

from mc.network import NetworkLayer as NL

from pygenn.genn_model import GeNNModel

from mc.defaults import default_neuron, default_synapse

model = GeNNModel("float","testmodel")

layer_def = {"layer_name":"testlayer",
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

nl = NL(model,**layer_def)

import pdb
pdb.set_trace()
