from genn_models.utils import convert_neuron_mod_data_cont_to_event

from ..settings import mod_type

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

post_plast_vars = ["r", "targ_mode"]

model_def = {
    "class_name": "input",
    "param_names": [],
    "var_name_types": [("r", "scalar"),
                       ("r_min_1", "scalar"),
                       ("r_trace", "scalar"),
                       ("r_trace_min_1", "scalar"),
                       ("r_trace_min_2", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY),
                       ("db", "scalar")],
    "sim_code": """
        $(r_min_1) = $(r);
        $(r) = $(Isyn);

        $(r_trace_min_2) = $(r_trace_min_1);
        $(r_trace_min_1) = $(r_trace);
        $(r_trace) += DT * ($(r_min_1) - $(r_trace));
    """
}

param_space = {}

var_space = {
    "r": 0.0,
    "r_min_1": 0.0,
    "r_trace": 0.0,
    "r_trace_min_1": 0.0,
    "r_trace_min_2": 0.0,
    "b": 0.0,
    "db": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}

TH = 0.1

if mod_type == "event":
    mod_dat = convert_neuron_mod_data_cont_to_event(mod_dat, post_plast_vars, th=TH)
