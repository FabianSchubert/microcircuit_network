'''
Initializations for weight update model variables.
'''

from pygenn.genn_model import init_var

wu_var_space_transmit_rate = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0})
}

wu_var_space_transmit_rate_diff = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
    "w_input": 0.0,
    "w_input_last": 0.0
}

wu_var_space_pp_basal = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
    "vbEff": 0.0,
    "dg": 0.0,
    "t_last": 0.0
}

wu_var_space_pinp = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
    "vbEff": 0.0,
    "dg": 0.0,
    "t_last": 0.0
}

wu_var_space_pp_apical = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0})
}

wu_var_space_ip = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
    "vEff": 0.0,
    "dg": 0.0,
    "t_last": 0.0
}

wu_var_space_pi = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
    "dg": 0.0,
    "t_last": 0.0
}

wu_var_space_ip_back = {
    "g": 1.0
}
