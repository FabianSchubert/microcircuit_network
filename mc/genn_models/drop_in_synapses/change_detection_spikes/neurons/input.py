from .utils import act_func, d_act_func

from genn_models.event_utils import convert_event_neur_dict, convert_event_neur_var_space_dict, convert_event_neur_param_space_dict

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

post_plast_vars = []

model_def = {
    "class_name": "input",
    "param_names": [],
    "var_name_types": [("r", "scalar"),
                       ("d_ra", "scalar"),
                       ("u", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY), ("db", "scalar")],
    "sim_code": f"""
        $(u) = $(Isyn);
        $(r) = $(u);
    """
}

model_def = convert_event_neur_dict(model_def, post_plast_vars)

param_space = {}

param_space = convert_event_neur_param_space_dict(param_space)

var_space = {
    "r": 0.0,
    "d_ra": 0.0,
    "u": 0.0,
    "b": 0.0,
    "db": 0.0
}

var_space = convert_event_neur_var_space_dict(var_space, post_plast_vars)

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}
