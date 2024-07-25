# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import types
import logging

from .llama_forward import forward_llama_model, forward_llama_for_causal_lm
from .mistral_forward import forward_mistral_model, forward_mistral_for_causal_lm


logger = logging.getLogger(__name__)


model_forward_functions = {
    'llama': forward_llama_model,
    'mistral': forward_mistral_model,
    'phi3': forward_mistral_model,
}
causal_lm_forward_functions = {
    'llama': forward_llama_for_causal_lm,
    'mistral': forward_mistral_for_causal_lm,
    'phi3': forward_mistral_for_causal_lm,
}


def replace_methods(model, model_type='llama'):
    if model_type not in model_forward_functions:
        logger.warning(f'Setting model type to llama for unrecognized model type: {model_type}')
        model_type = 'llama'
    model.forward = types.MethodType(causal_lm_forward_functions[model_type], model)
    model.model.forward = types.MethodType(model_forward_functions[model_type], model.model)
