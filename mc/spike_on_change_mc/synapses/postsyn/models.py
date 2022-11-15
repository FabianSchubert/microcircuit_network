'''
Create model instances of postsynaptic models from model
definitions.
'''

from pygenn.genn_model import create_custom_postsynaptic_class

from .model_defs import ps_model_integrator_update as ps_model_integrator_update_def

integrator_update = create_custom_postsynaptic_class(**ps_model_integrator_update_def)
