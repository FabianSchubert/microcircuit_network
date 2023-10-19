from .utils import act_func, d_act_func

from genn_models.utils import convert_neuron_mod_data_cont_to_event

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from ..settings import mod_type

post_plast_vars = ["err"]

model_def = {
    "class_name": "output",
    "param_names": [],
    "var_name_types": [("r", "scalar"),
                       ("r_targ", "scalar"),
                       ("err", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY),
                       ("db", "scalar")],
    "sim_code": """
        $(r) = $(Isyn_net) + $(b);
        $(r_targ) = $(Isyn);
        $(err) = $(r_targ) - $(r);
    """,
    "additional_input_vars": [("Isyn_net", "scalar", 0.0)]
}

param_space = {}

var_space = {
    "r": 0.0,
    "r_targ": 0.0,
    "err": 0.0,
    "b": 0.0,
    "db": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}

TH = 100.0

if mod_type == "event":
    mod_dat = convert_neuron_mod_data_cont_to_event(mod_dat, post_plast_vars, th=TH)
