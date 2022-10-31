step_current_model = {
	"name": "step_current",
	"injection_code" : """
        if ($(startStep) < ($(endStep) - 1) && $(t) >= $(stepTimes)[$(startStep) + 1]*DT) {
            $(startStep)++;
        }
        if($(t) >= $(stepTimes)[$(startStep)]) {
            $(injectCurrent, $(stepAmpls)[$(startStep)]);
        }
    """,
    "var_name_types": [
        ("startStep", "unsigned int"),
        ("endStep", "unsigned int")],
    "extra_global_params" : [("stepAmpls", "scalar*"), ("stepTimes", "int*")]
}