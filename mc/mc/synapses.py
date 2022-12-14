#! /usr/bin/env python3
import typing
from dataclasses import dataclass, field

from pygenn.genn_model import (create_custom_weight_update_class,
                               create_custom_postsynaptic_class,
                               init_connectivity, init_var)

from .utils import (merge_wu_def, merge_dicts,
                    merge_ps_def)

SCALE_WEIGHT_INIT = 0.5


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
    ps_model_transmit: dict = field(default_factory=dict)
    ps_model_plast: dict = field(default_factory=dict)
    ps_param_space_transmit: dict = field(default_factory=dict)
    ps_param_space_plast: dict = field(default_factory=dict)
    ps_var_space_transmit: dict = field(default_factory=dict)
    ps_var_space_plast: dict = field(default_factory=dict)
    connectivity_initialiser: "typing.Any" = None
    ps_target_var: str = "Isyn"
    norm_after_init: typing.Any = False

    def build_wu_model(self, plastic):
        '''
        Build either a "plastic" weight update
        model by merging the transmission and
        plastic weight update definition, or
        a static model using only the transmission
        model definition.
        '''
        if plastic:
            merged_wu_model_def = merge_wu_def("plastic_wu_model",
                                               self.w_update_model_transmit,
                                               self.w_update_model_plast)

            return create_custom_weight_update_class(**merged_wu_model_def)

        return create_custom_weight_update_class(**self.w_update_model_transmit)

    def build_ps_model(self, plastic):
        '''
        Build either a "plastic" postsynaptic
        model by merging the transmission and
        plastic postsynaptic model definition, or
        a static model using only the transmission
        model definition.
        '''
        if plastic:
            merged_ps_model_def = merge_ps_def("plastic_ps_model",
                                               self.ps_model_transmit,
                                               self.ps_model_plast)

            return create_custom_postsynaptic_class(**merged_ps_model_def)

        return create_custom_postsynaptic_class(**self.ps_model_transmit)

    def connect_pops(self, name, genn_model,
                     target, source, plastic=True):

        wu_model = self.build_wu_model(plastic)
        ps_model = self.build_ps_model(plastic)

        if plastic:
            wu_param_space = merge_dicts(self.wu_param_space_plast,
                                         self.wu_param_space_transmit)
            wu_var_space = merge_dicts(self.wu_var_space_plast,
                                       self.wu_var_space_transmit)

            wu_pre_var_space = merge_dicts(self.wu_pre_var_space_plast,
                                           self.wu_pre_var_space_transmit)
            wu_post_var_space = merge_dicts(self.wu_post_var_space_plast,
                                            self.wu_post_var_space_transmit)

            ps_param_space = merge_dicts(self.ps_param_space_plast,
                                         self.ps_param_space_transmit)
            ps_var_space = merge_dicts(self.ps_var_space_plast,
                                       self.ps_var_space_transmit)
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
            wu_post_var_space, ps_model,
            ps_param_space, ps_var_space,
            self.connectivity_initialiser
        )
        _syn_pop.ps_target_var = self.ps_target_var

        _syn_pop.norm_after_init = self.norm_after_init

        return _syn_pop


class SynapseDense(SynapseBase):
    '''
    SynapseDense extends the base synapse
    class by specifying the matrix type
    as "DENSE_INDIVIDUALG".
    '''

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.matrix_type = "DENSE_INDIVIDUALG"


class SynapseSparseOneToOne(SynapseBase):
    '''
    SynapseSparseOneToOne extends the
    base synapse class by specifying the
    matrix type as "SPARSE_INDIVIDUALG", which
    refers to a sparse matrix representation
    and an individual set of variables for
    every synapse in a population. Furthermore,
    the connectivity initialiser is set to be
    "OneToOne", meaning that this synapse class
    can only connect populations of the same size.
    '''

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.matrix_type = "SPARSE_INDIVIDUALG"
        self.connectivity_initialiser = init_connectivity("OneToOne", {})


class SynapsePPBasal(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        self.ps_target_var = "Isyn_vb"


class SynapsePINP(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        self.ps_target_var = "Isyn_vb"


class SynapsePPApical(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        self.ps_target_var = "Isyn_va_exc"


class SynapseIP(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)


class SynapseIPBack(SynapseSparseOneToOne):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ps_target_var = "u_td"


class SynapsePI(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        self.ps_target_var = "Isyn_va_int"
