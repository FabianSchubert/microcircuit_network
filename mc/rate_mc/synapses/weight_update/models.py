#! /usr/bin/env python3

from pygenn.genn_model import create_custom_weight_update_class

from .model_defs import (wu_model_pp_basal as wu_model_pp_basal_def,
						 wu_model_pp_apical as wu_model_pp_apical_def,
						 wu_model_pi as wu_model_pi_def,
						 wu_model_pi_back as wu_model_pi_back_def,
						 wu_model_ip as wu_model_ip_def,
						 wu_model_inpp as wu_model_inpp_def)

wu_model_pp_basal = create_custom_weight_update_class(**wu_model_pp_basal_def)

wu_model_pp_apical = create_custom_weight_update_class(**wu_model_pp_apical_def)

wu_model_inpp = create_custom_weight_update_class(**wu_model_inpp_def)

wu_model_pi = create_custom_weight_update_class(**wu_model_pi_def)

wu_model_pi_back = create_custom_weight_update_class(**wu_model_pi_back_def)

wu_model_ip = create_custom_weight_update_class(**wu_model_ip_def)

