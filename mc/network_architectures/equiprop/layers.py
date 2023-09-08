from pygenn.genn_model import create_custom_neuron_class

from network_base.layer import LayerBase

class InputLayer(LayerBase):

    def __init__(self, name, genn_model,
                 input_mod_dat, N, plastic=True,
                 optimizer_params={}):

        super().__init__(name, genn_model,
                         plastic)

        input_model_def = input_mod_dat["model_def"]
        input_param_space = input_mod_dat["param_space"]
        input_var_space = input_mod_dat["var_space"]

        _input_model = create_custom_neuron_class(**input_model_def)

        self.add_neur_pop("input_pop", N, _input_model,
                          input_param_space, input_var_space,
                          optimizer_params=optimizer_params)


class HiddenLayer(LayerBase):

    def __init__(self, name, genn_model,
                 hidden_mod_dat, N, plastic=True,
                 optimizer_params={}):

        super().__init__(name, genn_model,
                         plastic)

        hidden_model_def = hidden_mod_dat["model_def"]
        hidden_param_space = hidden_mod_dat["param_space"]
        hidden_var_space = hidden_mod_dat["var_space"]

        _hidden_model = create_custom_neuron_class(**hidden_model_def)

        self.add_neur_pop("hidden_pop", N, _hidden_model,
                          hidden_param_space, hidden_var_space,
                          optimizer_params=optimizer_params)


class OutputLayer(LayerBase):

    def __init__(self, name, genn_model,
                 output_mod_dat, N, plastic=True,
                 optimizer_params={}):

        super().__init__(name, genn_model,
                         plastic)

        output_model_def = output_mod_dat["model_def"]
        output_param_space = output_mod_dat["param_space"]
        output_var_space = output_mod_dat["var_space"]

        _output_model = create_custom_neuron_class(**output_model_def)

        self.add_neur_pop("output_pop", N, _output_model,
                          output_param_space, output_var_space,
                          optimizer_params=optimizer_params)
