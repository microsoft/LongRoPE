# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import types

from .llama_forward import forward_llama_model, forward_llama_for_causal_lm


model_forward_functions = {
    'llama': forward_llama_model,
    'mistral': forward_llama_model,
}
causal_lm_forward_functions = {
    'llama': forward_llama_for_causal_lm,
    'mistral': forward_llama_for_causal_lm,
}


def replace_methods(model, model_type='llama'):
    model.forward = types.MethodType(causal_lm_forward_functions[model_type], model)
    model.model.forward = types.MethodType(model_forward_functions[model_type], model.model)
