#! /usr/bin/env python3

from neuron.neurons import pyr_neuron, int_neuron
from synapse.synapses import (synapse_pp_basal,
                                synapse_pp_apical,
                                synapse_pi,s ynapse_ip)




rate_network = {
    
    "name": "default_net",
    "layers": [

    ]
}


spiking_network_params = {
    "name": "SpikeNet",

    "dim_input": 2,
    "dim_output": 1,
    "dim_hidden": [20],

    "dt": 0.1,

    "exc_model_def": {
        "class_name": "exc",
        "param_names": ["glk", "gb", "ga"],
        "var_name_types": [("u","scalar"), ("r", "scalar"),
                            ("va_int","scalar"),("va_exc","scalar"),
                            ("va","scalar"),
                            ("vb","scalar")],
        "additional_input_vars": [("Isyn_va_int","scalar",0.0),
                                ("Isyn_va_exc","scalar",0.0),
                                ("Isyn_vb","scalar",0.0)],
        "sim_code": f"""
                    $(vb) = $(Isyn_vb);
                    $(va_int) = $(Isyn_va_int);
                    $(va_exc) = $(Isyn_va_exc);
                    $(va) = $(va_int) + $(va_exc);
                    $(u) += DT * ( -($(glk)+$(gb)+$(ga))*$(u)
                    + $(gb)*$(vb)
                    + $(ga)*$(va) );
                    $(r) = {act_func('$(u)')};
                    //$(r) = $(u);
                    """,
        "threshold_condition_code": "$(gennrand_uniform) < $(r)*DT",
        "reset_code": None
    },

    "int_model_def": {
        "class_name": "int",
        "param_names": ["glk","gd","gsom"],
        "var_name_types": [("u","scalar"),("v","scalar"),("r","scalar")],
        "additional_input_vars": [("u_td","scalar",0.0)],
        "sim_code": f"""
                    $(v) = $(Isyn);
                    $(u) += DT * ( -$(glk)*$(u)
                    + $(gd)*( $(v)-$(u) )
                    + $(gsom)*( $(u_td) - $(u) ));
                    $(r) = {act_func('$(u)')};
                    //$(r) = (1.0+tanh(2.0*$(u)))/2.0;
                    """,
        "threshold_condition_code": "$(gennrand_uniform) < $(r)*DT",
        "reset_code": None
    },

    "input_model_def": {
        "class_name": "input",
        "param_names": None,
        "var_name_types": [("r","scalar"),("u","scalar")],
        "sim_code": "$(r) = $(u);",
        "threshold_condition_code": "$(gennrand_uniform) < $(r)*DT",
        "reset_code": None
    },

    "output_model_def": {

        "class_name": "output",
        "param_names": ["glk","gb","ga","gsom"],
        "var_name_types": [("u","scalar"),
                            ("vtrg","scalar"),("r","scalar"),
                            ("vb","scalar")],
        "sim_code": f"""
                    $(vb) = $(Isyn);
                    $(u) += DT * ( -($(glk)+$(gb)+$(ga))*$(u)
                    + $(gb)*$(vb)
                    + $(gsom)*( $(vtrg) - $(u) ));
                    $(r) = {act_func('$(u)')};
                    """,
        "threshold_condition_code": "$(gennrand_uniform) < $(r)*DT",
        "reset_code": None
    },

    "weight_update_model_exc_to_exc_fwd_def": {
        "class_name": "weight_update_model_exc_to_exc_fwd_def",
        "param_names": ["muPP"],
        "var_name_types": [("g", "scalar"),("vbEff","scalar")],
        "synapse_dynamics_code": f"""
            $(addToInSyn, $(g) * $(r_pre));
            $(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
            $(g) += DT * $(muPP) * $(r_pre)
                * ($(r_post) - {act_func('$(vbEff)')} );
        """,
        "is_pre_spike_time_required": False,
        "is_post_spike_time_required": False
    },


    "weight_update_model_exc_to_exc_back_def": {
        "class_name": "weight_update_model_exc_to_exc_back_def",
        "param_names": ["muPP"],
        "var_name_types": [("g", "scalar")],
        "synapse_dynamics_code": f"""
            $(addToInSyn, $(g) * $(r_pre));
            $(g) += DT * $(muPP) * $(r_pre)
                * ($(r_post) - $(va_exc_post));
        """,
        "is_pre_spike_time_required": False,
        "is_post_spike_time_required": False
    },

    "weight_update_model_exc_to_int_def": {
        "class_name": "weight_update_model_exc_to_int_def",
        "param_names": ["muIP"],
        "var_name_types": [("g", "scalar"),("vEff","scalar")],
        "synapse_dynamics_code": f"""
            $(addToInSyn, $(g) * $(r_pre));
            $(vEff) = $(v_post) * $(gd_post)/($(glk_post)+$(gd_post));
            $(g) += DT * $(muIP) * $(r_pre)
                * ($(r_post) - {act_func('$(vEff)')});
        """,
        "is_pre_spike_time_required": False,
        "is_post_spike_time_required": False
    },

    "weight_update_model_exc_to_int_back_def": {
        "class_name": "weight_update_model_exc_to_int_back_def",
        "var_name_types": [("g","scalar")],
        "synapse_dynamics_code": """
            $(addToInSyn, $(g) * $(r_pre));
        """,
        "is_pre_spike_time_required": False,
        "is_post_spike_time_required": False
    },

    "weight_update_model_int_to_exc_def": {
        "class_name": "weight_update_model_int_to_exc_def",
        "param_names": ["muPI"],
        "var_name_types": [("g", "scalar")],
        "synapse_dynamics_code": """
            $(addToInSyn, $(g) * $(r_pre));
            $(g) -= DT * $(muPI) * $(r_pre)
                * $(va_post);
        """,
        "is_pre_spike_time_required": False,
        "is_post_spike_time_required": False
    },



    "exc_model_init": {
        "parameters": {
            "glk": 0.1,
            "ga": 0.8,
            "gb": 1.0
        },
        "variables": {
            "u": 0.0,
            "va": 0.0,
            "va_int": 0.0,
            "va_exc": 0.0,
            "vb": 0.0,
            "r": 0.0
        }
    },

    "int_model_init": {
        "parameters": {
            "glk": 0.1,
            "gd": 1.0,
            "gsom": 0.8
        },
        "variables": {
            "u": 0.0,
            "v": 0.0,
            "r": 0.0
        }
    },

    "input_model_init":{
        "parameters": {},
        "variables": {"r": 0.0, "u": 0.0}
    },

    "output_model_init": {
        "parameters": {
            "glk": 0.1,
            "gb": 1.0,
            "ga": 0.0,
            "gsom": 0.8
        },
        "variables": {
            "u": 0.0,
            "vb": 0.0,
            "vtrg": 0.0,
            "r": 0.0
        }
    },

    "synapse_input_to_hidden_exc_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
            "vbEff": 0.0
        },
        "parameters": {
            "muPP": 6e-3
        }
    },


    "synapse_hidden_exc_to_hidden_int_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
            "vEff": 0.0
        },
        "parameters": {
            "muIP": 6e-3
        }

    },

    "synapse_hidden_int_to_hidden_exc_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0})
        },
        "parameters": {
            "muPI": 6e-3
        }
    },

    "synapse_hidden_exc_to_hidden_exc_fwd_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
            "vbEff": 0.0
        },
        "parameters": {
            "muPP": 6e-3
        }
    },

    "synapse_hidden_exc_to_hidden_exc_back_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0})
        },
        "parameters": {
            "muPP": 0.
        }
    },

    "synapse_hidden_exc_to_output_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
            "vbEff": 0.0
        },
        "parameters": {
            "muPP": 6e-3
        }
    },

    "synapse_output_to_hidden_exc_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0})
        },
        "parameters": {
            "muPP": 6e-3
        }

    },

    "synapse_hidden_exc_to_hidden_int_back_init": {
        "variables": {
            "g": 1.0# init_var("OneToOne", {"min": -1.0, "max": 1.0})
        },
        "parameters": {}
    },

    "synapse_output_to_hidden_int_init": {
        "variables": {
            "g": 1.0#init_connectivity("OneToOne",{})# init_var("OneToOne", {"min": -1.0, "max": 1.0})
        },
        "parameters": {}
    }
}

default_network_params = {
    "name": "Net",

    "dim_input": 2,
    "dim_output": 1,
    "dim_hidden": [20],

    "dt": 0.1,

    "exc_model_def": {
        "class_name": "exc",
        "param_names": ["glk", "gb", "ga"],
        "var_name_types": [("u","scalar"), ("r", "scalar"),
                            ("va_int","scalar"),("va_exc","scalar"),
                            ("va","scalar"),
                            ("vb","scalar")],
        "additional_input_vars": [("Isyn_va_int","scalar",0.0),
                                ("Isyn_va_exc","scalar",0.0),
                                ("Isyn_vb","scalar",0.0)],
        "sim_code": f"""
                    $(vb) = $(Isyn_vb);
                    $(va_int) = $(Isyn_va_int);
                    $(va_exc) = $(Isyn_va_exc);
                    $(va) = $(va_int) + $(va_exc);
                    $(u) += DT * ( -($(glk)+$(gb)+$(ga))*$(u)
                    + $(gb)*$(vb)
                    + $(ga)*$(va) );
                    $(r) = {act_func('$(u)')};
                    //$(r) = $(u);
                    """,
        "threshold_condition_code": None,
        "reset_code": None
    },

    "int_model_def": {
        "class_name": "int",
        "param_names": ["glk","gd","gsom"],
        "var_name_types": [("u","scalar"),("v","scalar"),("r","scalar")],
        "additional_input_vars": [("u_td","scalar",0.0)],
        "sim_code": f"""
                    $(v) = $(Isyn);
                    $(u) += DT * ( -$(glk)*$(u)
                    + $(gd)*( $(v)-$(u) )
                    + $(gsom)*( $(u_td) - $(u) ));
                    $(r) = {act_func('$(u)')};
                    //$(r) = (1.0+tanh(2.0*$(u)))/2.0;
                    """,
        "threshold_condition_code": None,
        "reset_code": None
    },

    "input_model_def": {
        "class_name": "input",
        "param_names": None,
        "var_name_types": [("r","scalar"),("u","scalar")],
        "sim_code": "$(r) = $(u);",
        "threshold_condition_code": None,
        "reset_code": None
    },

    "output_model_def": {

        "class_name": "output",
        "param_names": ["glk","gb","ga","gsom"],
        "var_name_types": [("u","scalar"),
                            ("vtrg","scalar"),("r","scalar"),
                            ("vb","scalar")],
        "sim_code": f"""
                    $(vb) = $(Isyn);
                    $(u) += DT * ( -($(glk)+$(gb)+$(ga))*$(u)
                    + $(gb)*$(vb)
                    + $(gsom)*( $(vtrg) - $(u) ));
                    $(r) = {act_func('$(u)')};
                    """,
        "threshold_condition_code": None,
        "reset_code": None
    },

    "weight_update_model_exc_to_exc_fwd_def": {
        "class_name": "weight_update_model_exc_to_exc_fwd_def",
        "param_names": ["muPP"],
        "var_name_types": [("g", "scalar"),("vbEff","scalar")],
        "synapse_dynamics_code": f"""
            $(addToInSyn, $(g) * $(r_pre));
            $(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
            $(g) += DT * $(muPP) * $(r_pre)
                * ($(r_post) - {act_func('$(vbEff)')} );
        """,
        "is_pre_spike_time_required": False,
        "is_post_spike_time_required": False
    },


    "weight_update_model_exc_to_exc_back_def": {
        "class_name": "weight_update_model_exc_to_exc_back_def",
        "param_names": ["muPP"],
        "var_name_types": [("g", "scalar")],
        "synapse_dynamics_code": f"""
            $(addToInSyn, $(g) * $(r_pre));
            $(g) += DT * $(muPP) * $(r_pre)
                * ($(r_post) - $(va_exc_post));
        """,
        "is_pre_spike_time_required": False,
        "is_post_spike_time_required": False
    },

    "weight_update_model_exc_to_int_def": {
        "class_name": "weight_update_model_exc_to_int_def",
        "param_names": ["muIP"],
        "var_name_types": [("g", "scalar"),("vEff","scalar")],
        "synapse_dynamics_code": f"""
            $(addToInSyn, $(g) * $(r_pre));
            $(vEff) = $(v_post) * $(gd_post)/($(glk_post)+$(gd_post));
            $(g) += DT * $(muIP) * $(r_pre)
                * ($(r_post) - {act_func('$(vEff)')});
        """,
        "is_pre_spike_time_required": False,
        "is_post_spike_time_required": False
    },

    "weight_update_model_exc_to_int_back_def": {
        "class_name": "weight_update_model_exc_to_int_back_def",
        "var_name_types": [("g","scalar")],
        "synapse_dynamics_code": """
            $(addToInSyn, $(g) * $(r_pre));
        """,
        "is_pre_spike_time_required": False,
        "is_post_spike_time_required": False
    },

    "weight_update_model_int_to_exc_def": {
        "class_name": "weight_update_model_int_to_exc_def",
        "param_names": ["muPI"],
        "var_name_types": [("g", "scalar")],
        "synapse_dynamics_code": """
            $(addToInSyn, $(g) * $(r_pre));
            $(g) -= DT * $(muPI) * $(r_pre)
                * $(va_post);
        """,
        "is_pre_spike_time_required": False,
        "is_post_spike_time_required": False
    },



    "exc_model_init": {
        "parameters": {
            "glk": 0.1,
            "ga": 0.8,
            "gb": 1.0
        },
        "variables": {
            "u": 0.0,
            "va": 0.0,
            "va_int": 0.0,
            "va_exc": 0.0,
            "vb": 0.0,
            "r": 0.0
        }
    },

    "int_model_init": {
        "parameters": {
            "glk": 0.1,
            "gd": 1.0,
            "gsom": 0.8
        },
        "variables": {
            "u": 0.0,
            "v": 0.0,
            "r": 0.0
        }
    },

    "input_model_init":{
        "parameters": {},
        "variables": {"r": 0.0, "u": 0.0}
    },

    "output_model_init": {
        "parameters": {
            "glk": 0.1,
            "gb": 1.0,
            "ga": 0.0,
            "gsom": 0.8
        },
        "variables": {
            "u": 0.0,
            "vb": 0.0,
            "vtrg": 0.0,
            "r": 0.0
        }
    },

    "synapse_input_to_hidden_exc_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
            "vbEff": 0.0
        },
        "parameters": {
            "muPP": 6e-3
        }
    },


    "synapse_hidden_exc_to_hidden_int_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
            "vEff": 0.0
        },
        "parameters": {
            "muIP": 6e-3
        }

    },

    "synapse_hidden_int_to_hidden_exc_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0})
        },
        "parameters": {
            "muPI": 6e-3
        }
    },

    "synapse_hidden_exc_to_hidden_exc_fwd_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
            "vbEff": 0.0
        },
        "parameters": {
            "muPP": 6e-3
        }
    },

    "synapse_hidden_exc_to_hidden_exc_back_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0})
        },
        "parameters": {
            "muPP": 0.
        }
    },

    "synapse_hidden_exc_to_output_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
            "vbEff": 0.0
        },
        "parameters": {
            "muPP": 6e-3
        }
    },

    "synapse_output_to_hidden_exc_init": {
        "variables": {
            "g": init_var("Uniform", {"min": -1.0, "max": 1.0})
        },
        "parameters": {
            "muPP": 6e-3
        }

    },

    "synapse_hidden_exc_to_hidden_int_back_init": {
        "variables": {
            "g": 1.0# init_var("OneToOne", {"min": -1.0, "max": 1.0})
        },
        "parameters": {}
    },

    "synapse_output_to_hidden_int_init": {
        "variables": {
            "g": 1.0#init_connectivity("OneToOne",{})# init_var("OneToOne", {"min": -1.0, "max": 1.0})
        },
        "parameters": {}
    }
}
