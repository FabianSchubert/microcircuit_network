from .utils import act_func, d_act_func

from genn_models.utils import convert_neuron_mod_data_cont_to_event

from pygenn.genn_model import init_var

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from ..settings import mod_type

post_plast_vars = ["err_fb", "dr_min_2"]

model_def = {
    "class_name": "leaky_integrator",
    "param_names": [],
    "var_name_types": [("h", "scalar"),
                       ("r_min_1", "scalar"),
                       ("r", "scalar"),
                       ("r_trace", "scalar"),
                       ("r_trace_min_1", "scalar"),
                       ("r_trace_min_2", "scalar"),
                       ("dr_min_2", "scalar"),
                       ("dr_min_1", "scalar"),
                       ("dr", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY),
                       ("db", "scalar"),
                       ("learning", "int"),
                       ("err_fb", "scalar")],
    "sim_code": f"""
        $(err_fb) = $(Isyn_err_fb);

        $(r_min_1) = $(r);

        $(r_trace_min_2) = $(r_trace_min_1);
        $(r_trace_min_1) = $(r_trace);
        $(r_trace) += DT * ($(r_min_1) - $(r_trace));

        $(dr_min_2) = $(dr_min_1);
        $(dr_min_1) = $(dr);

        $(h) += DT * ($(Isyn) + $(learning) * $(err_fb) + $(Isyn_input) + $(b) - $(h));
        $(r) = {act_func("$(h)")};
        $(dr) = {d_act_func("$(h)")};
    """,
    "additional_input_vars": [("Isyn_err_fb", "scalar", 0.0),
                              ("Isyn_input", "scalar", 0.0)]
}

param_space = {}

var_space = {
    "h": 0.5,
    "r_min_1": 0.0,
    "r": 0.0,
    "r_trace": 0.0,
    "r_trace_min_1": 0.0,
    "r_trace_min_2": 0.0,
    "dr_min_2": 1.0,
    "dr_min_1": 1.0,
    "dr": 1.0,
    "b": 0.1,
    "db": 0.0,
    "learning": 1,
    "err_fb": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}

TH = 0.1

if mod_type == "event":
    mod_dat = convert_neuron_mod_data_cont_to_event(mod_dat, post_plast_vars, th=TH)
