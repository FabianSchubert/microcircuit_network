from abc import ABC, abstractmethod
import typing
import numpy as np
from pygenn.genn_model import GeNNModel
from tqdm import tqdm

class NetworkBase(ABC):
    """
    The Network abstract base class defines the underlying framework
    for defining network models using the setup function.

    Args:
        name (str):

            Name of the network instance.

        dt (float):

            Size of simulation time step.

        n_batches (int):

            Batch size for the main simulation.

        n_batches_val (int):

            Batch size for validation / testing simulations.

        spike_buffer_size (int):

            Time steps in the spike recording buffer for
            the main simulation.
            If the simulation time exceeds this number
            of time steps, recorded spikes are overwritten
            starting from t=0.

        spike_buffer_size_val (int):

            Time steps in the spike recording buffer for
            validation / testing simulations.

        spike_rec_pops (list[str]):

            List of strings with names of neuron populations
            from which spikes are to be recorded during the
            main simulation.

        spike_rec_pops_val (list[str]):

            List of strings with names of neuron populations
            from which spikes are to be recorded during the
            validation / testing simulations.

        cuda_visible_devices (bool):

            Needs some further explanation
        
        plastic (bool):

            Specifies whether the network should be instantiated
            with active plasticity.
    """

    def __init__(self,
                 name: str,
                 dt: float,
                 n_batches: int,
                 n_batches_val: int,
                 spike_buffer_size: int,
                 spike_buffer_size_val: int,
                 spike_rec_pops: typing.List[str],
                 spike_rec_pops_val: typing.List[str],
                 *args,
                 cuda_visible_devices=False,
                 plastic=True,
                 **kwargs):

        self.name = name
        self.dt = dt
        self.n_batches = n_batches
        self.n_batches_val = n_batches_val
        self.spike_buffer_size = spike_buffer_size,
        self.spike_buffer_size_val = spike_buffer_size_val
        self.spike_rec_pops = spike_rec_pops
        self.spike_rec_pops_val = spike_rec_pops_val
        self.cuda_visible_devices = cuda_visible_devices
        self.plastic = plastic

        genn_kwargs = {}

        if self.cuda_visible_devices:
            from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL
            genn_kwargs["selectGPUByDeviceID"] = True
            genn_kwargs["deviceSelectMethod"] = DeviceSelect_MANUAL

        self.genn_model = GeNNModel("float", self.name,
                                    backend="CUDA", **kwargs)

        self.genn_model.batch_size = self.n_batches

        self.genn_model.dT = self.dt

        self.layers = {}

        self.setup(*args, **kwargs)

        ############################
        # set up the spike recording
        if self.spike_rec_pops is None:
            self.spike_rec_pops = []

        for _pop in self.spike_rec_pops:
            self.neur_pops[_pop].spike_recording_enabled = True

        ############################

        self.genn_model.build()

        self.genn_model.load(num_recording_timesteps=self.spike_buffer_size)

        self.norm_weights()

        # only if this instance is plastic, we create a static
        # twin of the network as a member variable of itself
        # (otherwise, calling create_static_twin() would lead
        # to an infinite recursive loop).

        if self.plastic:
            self.static_twin_net = self.create_static_twin(self, *args, **kwargs)

    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @classmethod
    def create_static_twin(cls, instance, *args, **kwargs):
        """create a non-plastic copy of the
        network by removing the plastic component
        in the weight update rules for the copy.
        The copy self.static_twin will be an
        instance of the network itself."""

        '''
        self.name = name
        self.dt = dt
        self.n_batches = n_batches
        self.n_batches_val = n_batches_val
        self.spike_buffer_size = spike_buffer_size,
        self.spike_buffer_size_val = spike_buffer_size_val
        self.spike_rec_pops = spike_rec_pops
        self.spike_rec_pops_val = spike_rec_pops_val
        self.cuda_visible_devices = cuda_visible_devices
        self.plastic = plastic
        '''

        return cls(f"static_twin_{instance.name}",
                   instance.dt,
                   instance.n_batches_val, None,
                   instance.spike_buffer_size_val, None,
                   instance.spike_rec_pops_val, None,
                   *args, cuda_visible_devices=instance.cuda_visible_devices,
                   plastic=False, **kwargs)

    def norm_weights(self):

        # normalize weights
        for synpop in self.syn_pops.values():
            norm = synpop.norm_after_init
            assert norm in (False, "lin", "lin_inv", "sqrt", "sqrt_inv"), \
                """Error: wrong synapse normalisation argument"""
            if norm:
                weightview = synpop.vars["g"].view
                synpop.pull_var_from_device("g")

                if norm == "lin":
                    normfact = synpop.src.size
                elif norm == "lin_inv":
                    normfact = synpop.trg.size
                elif norm == "sqrt":
                    normfact = np.sqrt(synpop.src.size)
                else:
                    normfact = np.sqrt(synpop.trg.size)

                weightview[:] = weightview[:] / normfact
                synpop.push_var_to_device("g")

    def reinitialize(self):

        self.genn_model.reinitialise()
        self.norm_weights()

        if self.plastic:
            self.static_twin_net.reinitialize()

    @property
    def neur_pops(self):
        """a shortcut reference to the genn_model
        neuron population dictionary"""

        return self.genn_model.neuron_populations

    @property
    def syn_pops(self):
        """a shortcut reference to the genn_model
        synapse population dictionary"""

        return self.genn_model.synapse_populations

    @property
    def cs_pops(self):
        """a shortcut reference to the genn_model
        current source dictionary"""

        return self.genn_model.current_sources

    @property
    def weights(self):

        weights = {}

        for key, syn_pop in self.syn_pops.items():
            view = syn_pop.vars["g"].view

            syn_pop.pull_var_from_device("g")

            # copy weights into new numpy array
            # (just storing a reference in the list could lead to
            # very unpredictable behavior)
            weights[key] = np.array(view)

        return weights

    @weights.setter
    def weights(self, weights):

        for key, value in weights.items():
            _syn_pop = self.syn_pops[key]

            _view = _syn_pop.vars["g"].view
            _view[:] = value

            _syn_pop.push_var_to_device("g")

    @property
    def biases(self):

        biases = {}

        for key, neur_pop in self.neur_pops.items():
            view = neur_pop.vars["b"].view

            neur_pop.pull_var_from_device("b")

            biases[key] = np.array(view)

        return biases

    @biases.setter
    def biases(self, biases):

        for key, value in biases.items():
            _neur_pop = self.neur_pops[key]

            _view = _neur_pop.vars["b"].view
            _view[:] = value

            _neur_pop.push_var_to_device("b")

    def add_layer(self, name, layertype, *args, **kwargs):
        assert name not in self.layers.keys(), \
            """Layer name already used"""
        self.layers[name] = layertype(name, self.genn_model, *args, **kwargs)

    def add_current_source(self, name, target, cs_model,
                           cs_params, cs_vars,
                           cs_extra_global_params):

        _ref = self.genn_model.add_current_source(name, cs_model,
                                                  self.neur_pops[target],
                                                  cs_params, cs_vars)

        for pkey, pval in cs_extra_global_params.items():
            _ref.set_extra_global_param(pkey, pval)

        return _ref

    def run_sim(self, T,
                ext_data_pop_vars, readout_neur_pop_vars,
                readout_syn_pop_vars,
                t_sign_validation=None,
                data_validation=None,
                show_progress=True,
                show_progress_val=True,
                NT_skip_batch_plast=1,
                force_self_pred_state=False,
                force_fb_align=False):
        """
        run_sim simulates the network and allows the user
        to provide input data as well as specify targets
        for readout/recording.

        Args:

            ext_data_pop_vars (list):

                        Additional data and time signatures targeting
                        neuron population variables in the network.

                        ext_data_pop_vars should be a list tuples of the form
                        (numpy_data, numpy_times, target_pop, target_var).

                        numpy_times is a list of time indices whose length
                        should be the same as the first dimension of
                        numpy_data.

            readout_neur_pop_vars (list):

                        List of tuples of the form
                        (readout_pop, readout_var, numpy_times).
                        readout_pop is the name of the population
                        to be read from & readout var specifies the
                        variable to be pulled.
                        numpy_times is an array specifying
                        the readout times.

            readout_syn_pop_vars (list):

                        Same structure as readout_neur_pop_vars.

            t_sign_validation (np.ndarray):

                        Time signatures for calling a validation run.

            data_validation (list):

                        List of dictionaries, each of which holds
                        arguments for a validation run executed at
                        one of the time steps given in t_sign_validation.
                        The arguments required are a subset of the arguments
                        provided to run_sim, namely
                        {T, ext_data_pop_vars, readout_neur_pop_vars}.
                        Note that this does not include any "target data",
                        e.g. for the readout. The validation runs only yield
                        the simulation data specified in readout_neur_pop_vars,
                        and it is the user's responsibility to construct
                        meaningful validation measures from the data that is
                        returned. Furthermore, there is no option for
                        recording synapses, since they are kept fixed
                        during the validation.

            show_progress (bool):

                        If true, a tqdm progress bar is displayed during
                        the simulation run.

            show_progress_val (bool):

                        Same as show_progress, but for the static twin.

            NT_skip_batch_plast (int):

                        Number of time steps between an update where
                        the synaptic and bias changes stored in the individual
                        batch instances in dg and db are reduced (averaged
                        over the batches) and added to the weights and biases.
                        After this, the synaptic change variables dg and the
                        bias change variables db are set back to zero.
                        The default value of 1 means that this happens every
                        time step (generally not recommended for performance).
        """

        NT = int(T / self.dt)

        input_views = []

        # List of indices storing the current "head" index from
        # which to load when the next time data is pushed to a population.
        idx_data_heads = []

        # t_sign = np.ndarray((0)) if t_sign is None else t_sign

        ext_data_pop_vars = [] if ext_data_pop_vars is None else ext_data_pop_vars

        n_inputs = len(ext_data_pop_vars)

        readout_neur_pop_vars = [] if readout_neur_pop_vars is None else readout_neur_pop_vars

        readout_syn_pop_vars = [] if readout_syn_pop_vars is None else readout_syn_pop_vars

        # List for copies of time signatures for extra input data.
        time_signatures_ext_data = []

        if data_validation:

            assert t_sign_validation is not None, \
                """Error: validation data was provided,
                but no time signatures were given.
                """

            assert t_sign_validation.shape[0] == len(data_validation), \
                """Error: Number of validation time signatures
                does not match number of validation datasets."""

            # List to store the results of the validation runs.
            results_validation = []

            # Counter for keeping track of the next validation run
            idx_validation_runs = 0

        for ext_dat, numpy_times, target_pop, target_var in ext_data_pop_vars:
            assert ext_dat.ndim == (2 if self.n_batches == 1 else 3), \
                """Input data array does
                not have appropriate dimensions.
                Needs to be 2 if n_batches = 1, else 3"""

            _target_pop = self.neur_pops[target_pop]
            assert ext_dat.shape[-1] == _target_pop.size, \
                """Input data size in last dimension and
                input population size do not match."""

            if self.n_batches > 1:
                assert ext_dat.shape[1] == self.n_batches, \
                    """Input data in second dimension and
                    batch size do not match."""

            assert len(numpy_times) > 0, \
                """Error: Passed empty input
                data time span list."""

            assert np.all(numpy_times[:-1] < numpy_times[1:]), \
                """Error: Input times are not correctly
                ordered"""

            assert numpy_times.shape[0] == ext_dat.shape[0], \
                """Error: Size of input data does not match
                number of time signatures."""

            time_signatures_ext_data.append(np.array(numpy_times))

            idx_data_heads.append(0)

            input_views.append(_target_pop.vars[target_var].view)

        readout_views = {}

        readout_neur_arrays = {}

        readout_syn_arrays = {}

        time_signatures_readout_neur_pop = []
        idx_readout_neur_pop_heads = []

        time_signatures_readout_syn_pop = []
        idx_readout_syn_pop_heads = []

        for readout_pop, readout_var, t_sign_readout in readout_neur_pop_vars:
            _dict_name = f'{readout_pop}_{readout_var}'
            _view = self.neur_pops[readout_pop].vars[readout_var].view
            readout_views[_dict_name] = _view
            readout_neur_arrays[_dict_name] = np.ndarray(
                (t_sign_readout.shape[0], self.n_batches, self.neur_pops[readout_pop].size))

            time_signatures_readout_neur_pop.append(np.array(t_sign_readout))
            idx_readout_neur_pop_heads.append(0)

        for readout_pop, readout_var, t_sign_readout in readout_syn_pop_vars:
            _dict_name = f'{readout_pop}_{readout_var}'
            _trg_size = self.syn_pops[readout_pop].trg.size
            _src_size = self.syn_pops[readout_pop].src.size
            readout_syn_arrays[_dict_name] = np.ndarray((t_sign_readout.shape[0],
                                                         _trg_size,
                                                         _src_size))

            time_signatures_readout_syn_pop.append(np.array(t_sign_readout))
            idx_readout_syn_pop_heads.append(0)

        ####################################################
        # prepare some internal state variables / parameters

        # reset some internal state variables
        self.genn_model.t = 0.0
        self.genn_model.timestep = 0

        # reset inSyn to zero, so that changes in the input
        # are correctly tracked.
        for _pop in self.syn_pops.values():
            _pop.in_syn = np.zeros(_pop.trg.size)
            _pop.push_in_syn_to_device()

        ####################################################

        for t in tqdm(range(NT), disable=not show_progress, leave=self.plastic):

            # manual variable manipulation

            self.push_ext_data(t, n_inputs, time_signatures_ext_data,
                               ext_data_pop_vars, input_views, idx_data_heads)

            # check if validation run should be
            # executed in this time step.
            if data_validation and self.plastic:
                if (idx_validation_runs < t_sign_validation.shape[0]
                        and t_sign_validation[idx_validation_runs] <= t*self.dt):

                    results_validation.append(
                        self.run_validation(**data_validation[idx_validation_runs],
                                            show_progress=show_progress_val))

                    idx_validation_runs += 1

            self.genn_model.step_time()

            if self.plastic and (t % NT_skip_batch_plast == 0):
                self.genn_model.custom_update("WeightChangeBatchReduce")
                self.genn_model.custom_update("BiasChangeBatchReduce")
                self.genn_model.custom_update("Plast")

                if force_fb_align:
                    self.align_fb_weights()
                if force_self_pred_state:
                    self.init_self_pred_state()

            self.pull_neur_var_data(t, readout_neur_pop_vars,
                                    time_signatures_readout_neur_pop,
                                    readout_neur_arrays, readout_views,
                                    idx_readout_neur_pop_heads)

            self.pull_syn_var_data(t, readout_syn_pop_vars,
                                   time_signatures_readout_syn_pop,
                                   readout_syn_arrays,
                                   idx_readout_syn_pop_heads)

        readout_spikes = {}

        if len(self.spike_rec_pops) > 0:
            self.genn_model.pull_recording_buffers_from_device()

            for _pop in self.spike_rec_pops:
                readout_spikes[_pop] = self.neur_pops[_pop].spike_recording_data

        if data_validation:
            return (readout_neur_arrays, readout_syn_arrays,
                    readout_spikes, results_validation)

        return readout_neur_arrays, readout_syn_arrays, readout_spikes

    def run_validation(self, T,
                       ext_data_pop_vars,
                       readout_neur_pop_vars, show_progress=True):
        '''
        This method runs a network simulation on the
        static twin after copying the current weight
        configuration to the twin. The output can
        be used e.g. to validate network performance.
        '''

        self.static_twin_net.genn_model.reinitialise()

        # store all the weights and biases in the network
        # and copy them over to the static twin

        self.static_twin_net.weights = self.weights
        self.static_twin_net.biases = self.biases

        result_neur_arrays, _, result_spikes = self.static_twin_net.run_sim(
            T, ext_data_pop_vars, readout_neur_pop_vars,
            [], show_progress=show_progress)

        return {"neur_var_rec": result_neur_arrays,
                "spike_rec": result_spikes}

    def push_ext_data(self, t, n_inputs, time_signatures_ext_data,
                      ext_data_pop_vars, input_views, idx_data_heads):

        for k in range(n_inputs):

            # if the array of time signatures is not empty...
            if time_signatures_ext_data[k].shape[0] > 0:

                # check if the current time is equal to the
                # current first element in the
                # array of time signatures.
                if time_signatures_ext_data[k][0] <= t * self.dt:
                    input_views[k][:] = ext_data_pop_vars[k][0][idx_data_heads[k]]

                    self.neur_pops[ext_data_pop_vars[k][2]].push_var_to_device(
                        ext_data_pop_vars[k][3])

                    idx_data_heads[k] += 1

                    # remove the current first element of the
                    # time signatures after use.
                    time_signatures_ext_data[k] = time_signatures_ext_data[k][1:]

    def pull_neur_var_data(self, t, readout_neur_pop_vars,
                           time_signatures_readout_neur_pop,
                           readout_neur_arrays, readout_views,
                           idx_readout_neur_pop_heads):

        for k, (readout_pop, readout_var, _) in enumerate(readout_neur_pop_vars):

            if time_signatures_readout_neur_pop[k].shape[0] > 0:

                if time_signatures_readout_neur_pop[k][0] <= t*self.dt:
                    self.neur_pops[readout_pop].pull_var_from_device(
                        readout_var)
                    _dict_name = f'{readout_pop}_{readout_var}'
                    readout_neur_arrays[_dict_name][idx_readout_neur_pop_heads[k]
                    ] = readout_views[_dict_name]

                    idx_readout_neur_pop_heads[k] += 1

                    time_signatures_readout_neur_pop[k] = time_signatures_readout_neur_pop[k][1:]

    def pull_syn_var_data(self, t, readout_syn_pop_vars,
                          time_signatures_readout_syn_pop,
                          readout_syn_arrays,
                          idx_readout_syn_pop_heads):

        for k, (readout_pop, readout_var, _) in enumerate(readout_syn_pop_vars):

            if time_signatures_readout_syn_pop[k].shape[0] > 0:

                if time_signatures_readout_syn_pop[k][0] <= t*self.dt:
                    self.syn_pops[readout_pop].pull_var_from_device(
                        readout_var)
                    _dict_name = f'{readout_pop}_{readout_var}'

                    readout_syn_arrays[_dict_name][idx_readout_syn_pop_heads[k]] = np.reshape(
                        self.syn_pops[readout_pop].get_var_values(
                            readout_var),
                        readout_syn_arrays[_dict_name].shape[1:], order='F')

                    idx_readout_syn_pop_heads[k] += 1

                    time_signatures_readout_syn_pop[k] = time_signatures_readout_syn_pop[k][1:]

