from pygenn.genn_model import create_custom_current_source_class

test_source = create_custom_current_source_class(
    "test_source",
    param_names=["amplitude"],
    injection_code="""
        $(injectCurrent, ($(gennrand_uniform) * 2.0 - 1.0) * $(amplitude));
    """
)

step_source_model = create_custom_current_source_class(
    "step_source",
    param_names=["n_samples_set", "pop_size", "batch_size",
                 "input_id_list_size", "input_times_list_size"],
    var_name_types=[("idx", "int"), ("t_next", "scalar")],
    extra_global_params=[("data", "scalar*"),
                         ("input_id_list", "int*"),
                         ("input_times_list", "scalar*")],
    injection_code="""
        // increase idx every time t surpasses the current next time to change
        // to a new pattern, except if we are at the end of the times list.

        if($(idx) < ($(input_times_list_size) - 1)){
            if($(t) >= $(input_times_list)[$(idx)+1]){
                $(idx)++;
            }
        }

        // input_id_list contains indices for samples to be drawn
        // from the dataset, and these are drawn concurrently
        // in batches. It starts back from the beginning of the list
        // if the list length is exceeded. This means that
        // one batch can contain samples corresponding to
        // indices from both the end and the start of sample_lst if
        // the length of sample_lst is not divisible by the batch size.

        const int sample_id = $(input_id_list)[($(idx)*int($(batch_size))+$(batch))%int($(input_id_list_size))];

        // the original data is of size (n_sample, pop_size)
        // and is flattened c-style.
        const int data_id = $(pop_size) * sample_id + $(id);
        $(injectCurrent, $(data)[data_id]);

        // if you want to simulate multiple epochs,
        // sample_lst should contain randomly shuffled versions
        // of [0,...,n_samples_set-1], concatenated together.
        // If you don't care about repeating the order of samples
        // in each epoch, you can also just provide one shuffled version of
        // [0,...,n_samples_set-1] (or even less if you are subsampling).
    """
)


