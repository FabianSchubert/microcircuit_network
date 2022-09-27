#! /usr/bin/env python3

from .weight_update.models import (wu_model_pp_basal, 
                                            wu_model_pp_apical,
                                            wu_model_pi,
                                            wu_model_ip)
from .weight_update.params import (wu_param_space_pp_basal, 
                                            wu_param_space_pp_apical,
                                            wu_param_space_pi,
                                            wu_param_space_ip)
from .weight_update.var_inits import (wu_var_space_pp_basal,
                                            wu_var_space_pp_apical,
                                            wu_var_space_pi,
                                            wu_var_space_ip)

# Entries set to None should be set individually,
synapse_dense = {
    "w_update_model": None,
    "matrix_type": "DENSE_INDIVIDUALG",
    "delay_steps": 0,
    "wu_param_space": None,
    "wu_var_space": None,
    "wu_pre_var_space": None,
    "w_post_var_space": None,
    "postsyn_model": "DeltaCurr",
    "ps_param_space": {},
    "ps_var_space": {},
    "connectivity_initialiser": None
}

# Pyr to pyr Basal Fwd
synapse_pp_basal = dict(synapse_dense)
synapse_pp_basal["w_update_model"] = wu_model_pp_basal
synapse_pp_basal["wu_param_space"] = wu_param_space_pp_basal
synapse_pp_basal["wu_var_space"] = wu_var_space_pp_basal
synapse_pp_basal["ps_target_var"] = "Isyn_vb"

# Pyr to pyr Apical Back
synapse_pp_apical = dict(synapse_dense)
synapse_pp_apical["w_update_model"] = wu_model_pp_apical
synapse_pp_apical["wu_param_space"] = wu_param_space_pp_apical
synapse_pp_apical["wu_var_space"] = wu_var_space_pp_apical
synapse_pp_apical["ps_target_var"] = "Isyn_va_exc"

# Pyr to int within layer
synapse_pi = dict(synapse_dense)
synapse_pi["w_update_model"] = wu_model_pi
synapse_pi["wu_param_space"] = wu_param_space_pi
synapse_pi["wu_var_space"] = wu_var_space_pi

# Int to pyr within layer
synapse_ip = dict(synapse_dense)
synapse_ip["w_update_model"] = wu_model_ip
synapse_ip["wu_param_space"] = wu_param_space_ip
synapse_ip["wu_var_space"] = wu_var_space_ip
synapse_ip["ps_target_var"] = "Isyn_va_int"