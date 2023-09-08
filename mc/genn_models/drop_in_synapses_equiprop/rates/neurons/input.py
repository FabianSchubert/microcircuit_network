from genn_models.utils import convert_neuron_mod_data_cont_to_event

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from ..settings import mod_type

post_plast_vars = []

model_def = {
    "class_name": "input",
    "param_names": ["beta"],
    "var_name_types": [("r", "scalar"),
                       ("u", "scalar"),
                       ("targ_mode", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY), ("db", "scalar")],
    "sim_code": """
        $(u) = $(Isyn);
        $(r) = $(u);
    """
}

param_space = {"beta": 1.0}

var_space = {
    "r": 0.0,
    "u": 0.0,
    "targ_mode": 0.0,
    "b": 0.0,
    "db": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}

if mod_type == "event":
    mod_dat = convert_neuron_mod_data_cont_to_event(mod_dat, post_plast_vars)
