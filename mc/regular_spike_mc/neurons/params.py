'''
Neuron model parameters stored in
dicts.
'''

pyr_hidden_param_space = {
    "glk": 0.1,
    "ga": 0.8,
    "gb": 1.0,
    "sigm_noise": 0.0,
    "spike_interv": 1
}

output_param_space = {
    "glk": 0.1,
    "gb": 1.0,
    "ga": 0.0,
    "sigm_noise": 0.0,
    "pop_size": None,
    "spike_interv": 1
}

int_param_space = {
    "glk": 0.1,
    "gd": 1.0,
    "gsom": 0.8,
    "spike_interv": 1
}

input_param_space = {
    "pop_size": None,
    "spike_interv": 1
}
