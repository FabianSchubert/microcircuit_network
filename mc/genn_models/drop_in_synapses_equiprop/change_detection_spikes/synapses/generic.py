from genn_models.utils import (WU_TRANSMIT, WU_VAR_SPACE_TRANSMIT,
                               WU_PARAM_SPACE_TRANSMIT,
                               PS_TRANSMIT, PS_VAR_SPACE_TRANSMIT,
                               PS_PARAM_SPACE_TRANSMIT,
                               generate_plast_wu_dict,
                               WU_PARAM_SPACE_PLAST, WU_VAR_SPACE_PLAST)

from pygenn.genn_model import init_var

from ..settings import mod_type

# reinit a dict instance, otherwise there is
# a risk of overwriting other stuff
w_update_model_transmit = dict(WU_TRANSMIT[mod_type])
wu_param_space_transmit = dict(WU_PARAM_SPACE_TRANSMIT[mod_type])
wu_var_space_transmit = dict(WU_VAR_SPACE_TRANSMIT[mod_type])

WEIGHT_SCALE = 1.0
wu_var_space_transmit["g"] = init_var("Uniform", {"min": -WEIGHT_SCALE, "max": WEIGHT_SCALE})

f = "(2.*$(targ_mode_post)-1.) * $(r_pre) * $(r_post) / $(beta)"
# 1 if targ_mode_post == 1, -1 if targ_mod_post == 0

wu_param_space_plast = dict(WU_PARAM_SPACE_PLAST[mod_type])
wu_param_space_plast["beta"] = 1.0
w_update_model_plast = generate_plast_wu_dict(mod_type, "weight_update_model_pyr_to_int", f,
                                              params=list(wu_param_space_plast.keys()))

wu_var_space_plast = dict(WU_VAR_SPACE_PLAST[mod_type])

ps_model_transmit = dict(PS_TRANSMIT[mod_type])
ps_param_space_transmit = dict(PS_PARAM_SPACE_TRANSMIT[mod_type])
ps_var_space_transmit = dict(PS_VAR_SPACE_TRANSMIT[mod_type])

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
