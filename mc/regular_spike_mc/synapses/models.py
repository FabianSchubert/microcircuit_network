#! /usr/bin/env python3

from pygenn.genn_model import create_custom_weight_update_class

from .postsyn.models import integrator_update

from .weight_update.model_defs import (wu_model_transmit_rate_diff,
                                       wu_model_pp_basal,
                                       wu_model_pp_apical,
                                       wu_model_pinp,
                                       wu_model_ip,
                                       wu_model_ip_back,
                                       wu_model_pi)

from .weight_update.params import (wu_param_space_transmit_rate,
                                   wu_param_space_pp_basal,
                                   wu_param_space_pp_apical,
                                   wu_param_space_ip,
                                   wu_param_space_pi,
                                   wu_param_space_ip_back,
                                   wu_param_space_pinp)

from pygenn.genn_model import init_var

from pygenn.genn_model import init_connectivity

from dataclasses import dataclass, field

from ..utils import merge_wu_def, merge_dicts  # _consistent


@dataclass
class SynapseBase:
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
    postsyn_model: "typing.Any" = integrator_update
    ps_param_space_transmit: dict = field(default_factory=dict)
    ps_param_space_plast: dict = field(default_factory=dict)
    ps_var_space_transmit: dict = field(default_factory=dict)
    ps_var_space_plast: dict = field(default_factory=dict)
    connectivity_initialiser: "typing.Any" = None
    ps_target_var: str = "Isyn"

    def build_static_wu_model(self):

        return create_custom_weight_update_class(**self.w_update_model_transmit)

    def build_plastic_wu_model(self):

        merged_wu_model_def = merge_wu_def("plastic_wu_model",
                                           self.w_update_model_transmit,
                                           self.w_update_model_plast)

        return create_custom_weight_update_class(**merged_wu_model_def)

    def connect_pops(self, name, genn_model, target, source, plastic=True):

        wu_model = self.build_plastic_wu_model() if plastic else self.build_static_wu_model()

        if plastic:
            wu_param_space = merge_dicts(self.wu_param_space_transmit,
                                         self.wu_param_space_plast)
            wu_var_space = merge_dicts(self.wu_var_space_transmit,
                                       self.wu_var_space_plast)
            wu_pre_var_space = merge_dicts(self.wu_pre_var_space_transmit,
                                           self.wu_pre_var_space_plast)
            wu_post_var_space = merge_dicts(self.wu_post_var_space_transmit,
                                            self.wu_post_var_space_plast)
            ps_param_space = merge_dicts(self.ps_param_space_transmit,
                                         self.ps_param_space_plast)
            ps_var_space = merge_dicts(self.ps_var_space_transmit,
                                       self.ps_var_space_plast)
        else:
            wu_param_space = dict(self.wu_param_space_transmit)
            wu_var_space = dict(self.wu_var_space_transmit)
            wu_pre_var_space = dict(self.wu_pre_var_space_transmit)
            wu_post_var_space = dict(self.wu_post_var_space_transmit)
            ps_param_space = dict(self.ps_param_space_transmit)
            ps_var_space = dict(self.ps_var_space_transmit)

        _syn_pop = genn_model.add_synapse_population(
            name, self.matrix_type, self.delay_steps,
            source, target, wu_model, wu_param_space,
            wu_var_space, wu_pre_var_space,
            wu_post_var_space, self.postsyn_model,
            ps_param_space, ps_var_space,
            self.connectivity_initialiser
        )
        _syn_pop.ps_target_var = self.ps_target_var

        return _syn_pop


class SynapseDense(SynapseBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.matrix_type = "DENSE_INDIVIDUALG"


class SynapseSparseOneToOne(SynapseBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.matrix_type = "SPARSE_GLOBALG"
        self.connectivity_initialiser = init_connectivity("OneToOne", {})


class SynapsePPBasal(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        N_norm = args[-1]

        self.w_update_model_transmit = wu_model_transmit_rate_diff
        self.w_update_model_plast = wu_model_pp_basal

        self.wu_param_space_transmit = wu_param_space_transmit_rate
        self.wu_param_space_plast = wu_param_space_pp_basal

        self.wu_var_space_transmit = {
            "g": init_var("Uniform",
                          {"min": -1./N_norm**.5, "max": 1./N_norm**.5})
        }

        self.wu_var_space_plast = {
            "g": init_var("Uniform",
                          {"min": -1./N_norm**.5, "max": 1./N_norm**.5}),
            "vbEff": 0.0,
            "dg": 0.0
        }

        self.ps_target_var = "Isyn_vb"


class SynapsePINP(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        N_norm = args[-1]

        self.w_update_model_transmit = wu_model_transmit_rate_diff
        self.w_update_model_plast = wu_model_pinp

        self.wu_param_space_transmit = wu_param_space_transmit_rate
        self.wu_param_space_plast = wu_param_space_pinp

        self.wu_var_space_transmit = {
            "g": init_var("Uniform",
                          {"min": -1./N_norm**.5, "max": 1./N_norm**.5})
        }

        self.wu_var_space_plast = {
            "g": init_var("Uniform",
                          {"min": -1./N_norm**.5, "max": 1./N_norm**.5}),
            "vbEff": 0.0,
            "dg": 0.0
        }

        self.ps_target_var = "Isyn_vb"


class SynapsePPApical(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        N_norm = args[-1]

        self.w_update_model_transmit = wu_model_transmit_rate_diff
        self.w_update_model_plast = wu_model_pp_apical

        self.wu_param_space_transmit = wu_param_space_transmit_rate
        self.wu_param_space_plast = wu_param_space_pp_apical

        self.wu_var_space_transmit = {
            "g": init_var("Uniform",
                          {"min": -2./N_norm**.5, "max": 2./N_norm**.5})
        }

        self.wu_var_space_plast = {
            "g": init_var("Uniform",
                          {"min": -2./N_norm**.5, "max": 2./N_norm**.5})
        }

        self.ps_target_var = "Isyn_va_exc"


class SynapseIP(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        N_norm = args[-1]

        self.w_update_model_transmit = wu_model_transmit_rate_diff
        self.w_update_model_plast = wu_model_ip

        self.wu_param_space_transmit = wu_param_space_transmit_rate
        self.wu_param_space_plast = wu_param_space_ip

        self.wu_var_space_transmit = {
            "g": init_var("Uniform",
                          {"min": -1./N_norm**.5, "max": 1./N_norm**.5})
        }

        self.wu_var_space_plast = {
            "g": init_var("Uniform",
                          {"min": -1./N_norm**.5, "max": 1./N_norm**.5}),
            "vEff": 0.0,
            "dg": 0.0
        }


class SynapseIPBack(SynapseSparseOneToOne):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.w_update_model_transmit = wu_model_transmit_rate_diff
        self.w_update_model_plast = wu_model_ip_back

        self.wu_param_space_transmit = wu_param_space_transmit_rate
        self.wu_param_space_plast = wu_param_space_ip_back

        self.wu_var_space_transmit = {
            "g": 1.0
        }
        self.wu_var_space_plast = {
            "g": 1.0
        }

        self.ps_target_var = "u_td"


class SynapsePI(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        N_norm = args[-1]

        self.w_update_model_transmit = wu_model_transmit_rate_diff
        self.w_update_model_plast = wu_model_pi

        self.wu_param_space_transmit = wu_param_space_transmit_rate
        self.wu_param_space_plast = wu_param_space_pi

        self.wu_var_space_transmit = {
            "g": init_var("Uniform",
                          {"min": -1./N_norm**.5, "max": 1./N_norm**.5})
        }

        self.wu_var_space_plast = {
            "g": init_var("Uniform",
                          {"min": -1./N_norm**.5, "max": 1./N_norm**.5}),
            "dg": 0.0
        }

        self.ps_target_var = "Isyn_va_int"
