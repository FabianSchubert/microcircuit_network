from network_base.layer import LayerBase

from pygenn.genn_model import create_custom_neuron_class

class HiddenLayer(LayerBase):
    def __init__(self, name, genn_model,
                 neur_mod_dat, rec_synapse_def,
                 Nneur, plastic=True, read_only_weights=False,
                 optimizer_params={}):
        super().__init__(
            name, genn_model,
            plastic, read_only_weights)

        neur_model_def = neur_mod_dat["model_def"]
        neur_param_space = neur_mod_dat["param_space"]
        neur_var_space = neur_mod_dat["var_space"]
        _neur_model = create_custom_neuron_class(**neur_model_def)

        self.add_neur_pop("hidden_pop", Nneur, _neur_model,
                          neur_param_space, neur_var_space,
                          optimizer_params=optimizer_params)

class InputLayer(LayerBase):
    def __init__(self, name, genn_model,
                 neur_mod_dat, Nneur,
                 plastic=False, read_only_weights=False,
                 optimizer_params={}):
        super().__init__(
            name, genn_model,
            plastic, read_only_weights)

        neur_model_def = neur_mod_dat["model_def"]
        neur_param_space = neur_mod_dat["param_space"]
        neur_var_space = neur_mod_dat["var_space"]
        _neur_model = create_custom_neuron_class(**neur_model_def)

        self.add_neur_pop("input_pop", Nneur, _neur_model,
                          neur_param_space, neur_var_space,
                          optimizer_params=optimizer_params)


class OutputLayer(LayerBase):
    def __init__(self, name, genn_model,
                 neur_mod_dat, Nneur,
                 plastic=True, read_only_weights=False,
                 optimizer_params={}):
        super().__init__(
            name, genn_model,
            plastic, read_only_weights)

        neur_model_def = neur_mod_dat["model_def"]
        neur_param_space = neur_mod_dat["param_space"]
        neur_var_space = neur_mod_dat["var_space"]
        _neur_model = create_custom_neuron_class(**neur_model_def)

        self.add_neur_pop("output_pop", Nneur, _neur_model,
                          neur_param_space, neur_var_space,
                          optimizer_params=optimizer_params)
