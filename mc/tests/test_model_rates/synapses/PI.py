'''
matrix_type: str = None  # = "DENSE_INDIVIDUALG"
    delay_steps: int = 0
    w_update_model_transmit: dict = field(default_factory=dict)
    w_update_model_plast: dict = field(default_factory=dict)
    wu_param_space_transmit: dict = field(default_factory=dict)
    wu_param_space_plast: dict = field(default_factory=dict)
    wu_var_space_transmit: dict = field(default_factory=dict)
    wu_var_space_plast: dict = field(default_factory=dict)
    wu_pre_var_space_transmit: dict = field(default_factory=dict)
    wu_pre_var_space_plast: dict = field(default_factory=dict)
    wu_post_var_space_transmit: dict = field(default_factory=dict)
    wu_post_var_space_plast: dict = field(default_factory=dict)
    ps_model_transmit: dict = field(default_factory=dict)
    ps_model_plast: dict = field(default_factory=dict)
    ps_param_space_transmit: dict = field(default_factory=dict)
    ps_param_space_plast: dict = field(default_factory=dict)
    ps_var_space_transmit: dict = field(default_factory=dict)
    ps_var_space_plast: dict = field(default_factory=dict)
    connectivity_initialiser: "typing.Any" = None
    ps_target_var: str = "Isyn"
'''

from ..utils import act_func
from pygenn.genn_model import init_var

w_update_model_transmit = {
    "class_name": "weight_update_model_transmit_rate",
    "param_names": [],
    "var_name_types": [("g", "scalar")],
    "synapse_dynamics_code": "$(addToInSyn, $(g) * $(r_pre));"
}

w_update_model_plast = {
    "class_name": "weight_update_model_int_to_pyr_def",
    "param_names": ["muPI", "tau"],
    "var_name_types": [("g", "scalar"), ("dg", "scalar")],
    "synapse_dynamics_code": f"""
        // SIM CODE PI
        $(dg) += DT * (- $(muPI) * $(r_pre) * $(va_post) - $(dg)) / $(tau);
        //$(g) += DT * $(muPI) * $(dg);
    """,
    "is_prev_pre_spike_time_required": False,
    "is_prev_post_spike_time_required": False,
    "is_pre_spike_time_required": False,
    "is_post_spike_time_required": False
}

wu_param_space_transmit = {}

wu_param_space_plast = {"muPI": 8*5e-4,
                     "tau": 30.}

wu_var_space_transmit = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0})
}

wu_var_space_plast = {
    "dg": 0.0
}

ps_model_transmit = {
    "class_name": "delta_update",
    "apply_input_code": "$(Isyn) += $(inSyn); $(inSyn) = 0.0;"
}

ps_param_space_transmit = {}

ps_var_space_transmit = {}

mod_dat = {
    "w_update_model_transmit": w_update_model_transmit,
    "w_update_model_plast": w_update_model_plast,
    "wu_param_space_transmit": wu_param_space_transmit,
    "wu_param_space_plast": wu_param_space_plast,
    "wu_var_space_transmit": wu_var_space_transmit,
    "wu_var_space_plast": wu_var_space_plast,
    "ps_model_transmit": ps_model_transmit,
    "ps_param_space_transmit": ps_param_space_transmit,
    "ps_var_space_transmit": ps_var_space_transmit,
    "norm_after_init": "sqrt"
}