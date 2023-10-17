from .utils import act_func, d_act_func

from genn_models.utils import convert_neuron_mod_data_cont_to_event

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from ..settings import mod_type

post_plast_vars = ["r", "targ_mode"]

model_def = {
    "class_name": "hidden",
    "param_names": ["eps", "beta"],
    "var_name_types": [("r", "scalar"),
                       ("u", "scalar"),
                       ("targ_mode", "scalar"),
                       ("r_targ", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY), ("db", "scalar")],
    "sim_code": f"""
        $(r_targ) = $(Isyn);
        $(u) = ($(u)
                + DT * $(eps) * (
                      {d_act_func("$(u)")} * ($(Isyn_regular) + $(b)) - $(u)
                    + $(beta) * $(targ_mode) * ($(r_targ) - $(r))
                    )
                );

        //$(u) = min(1.0, max($(u), 0.0));

        $(r) = {act_func("$(u)")};
    """,
    "additional_input_vars": [("Isyn_regular", "scalar", 0.0)]
}

param_space = {
    "eps": 0.5,
    "beta": .1
}

var_space = {
    "r": 0.0,
    "u": 0.0,
    "targ_mode": 0.0,
    "r_targ": 0.0,
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
