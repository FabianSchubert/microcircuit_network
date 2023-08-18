from genn_models.event_utils import (WU_TRANSMIT, WU_VAR_SPACE_TRANSMIT, WU_PARAM_SPACE_TRANSMIT,
                     PS_TRANSMIT, PS_VAR_SPACE_TRANSMIT, PS_PARAM_SPACE_TRANSMIT,
                     generate_event_plast_wu_dict, WU_PARAM_SPACE_PLAST, WU_VAR_SPACE_PLAST)

from pygenn.genn_model import init_var

# reinit a dict instance, otherwise there is
# a risk of overwriting other stuff
w_update_model_transmit = dict(WU_TRANSMIT)
wu_param_space_transmit = dict(WU_PARAM_SPACE_TRANSMIT)
wu_var_space_transmit = dict(WU_VAR_SPACE_TRANSMIT)

wu_var_space_transmit["g"] = 1.0

f = None

w_update_model_plast = generate_event_plast_wu_dict("weight_update_model_pyr_to_int_back", f)
wu_param_space_plast = dict(WU_PARAM_SPACE_PLAST)
wu_var_space_plast = dict(WU_VAR_SPACE_PLAST)

ps_model_transmit = dict(PS_TRANSMIT)
ps_param_space_transmit = dict(PS_PARAM_SPACE_TRANSMIT)
ps_var_space_transmit = dict(PS_VAR_SPACE_TRANSMIT)

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
    "norm_after_init": False
}
