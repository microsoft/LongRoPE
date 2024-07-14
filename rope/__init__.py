# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging

import torch
import numpy as np

from transformers import AutoConfig, AutoModelForCausalLM

from .longrope import LongRoPEScaledRotaryEmbedding, MixedLongRoPEScaledRotaryEmbedding, DynamicLongRoPEScaledRotaryEmbedding
from .yarn import YaRNScaledRotaryEmbedding
from utils.save_memory import replace_methods


logger = logging.getLogger(__name__)


def replace_rope(
    model: AutoModelForCausalLM,
    rope_class: type,
    rope_args: dict,
):
    """
    Replaces the RoPE module in each layer of the given model.

    Args:
        model (AutoModelForCausalLM): The model to modify.
        rope_class (type): The class of the new RoPE.
        rope_args (dict): The arguments to initialize the new RoPE.

    Returns:
        AutoModelForCausalLM: The modified model with the new RoPE in each layer.
    """
    for idx, layer in enumerate(model.model.layers):
        layer_rope_args = {}
        for k, v in rope_args.items():
            if type(v) is np.ndarray and v.ndim == 2:
                layer_rope_args[k] = v[idx]
            else:
                layer_rope_args[k] = v
        if 'dim' not in layer_rope_args:
            layer_rope_args['dim'] = layer.self_attn.head_dim
        device = "auto"
        if hasattr(layer.self_attn, "o_proj"):
            device = layer.self_attn.o_proj.weight.device
        elif hasattr(layer.self_attn, "dense"):
            device = layer.self_attn.dense.weight.device
        layer_rope_args['device'] = device
        layer.self_attn.rotary_emb = rope_class(**layer_rope_args)
    return model


def load_model(
    model_name_or_path: str,
    rope_method: str,
    max_position_embeddings: int = None,
    model_class: type = AutoModelForCausalLM,
    config: AutoConfig = None,
    rope_params: dict = None,
    attn_sliding_window: int = -1,
    save_memory: bool = False,
    **model_args,
):
    """
    Load a model and replace the RoPE module if required.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model.
        rope_method (str): The RoPE method to use. Possible values are 'pi', 'dy_ntk', 'yarn', 'yarn_dynamic',
            'longrope', 'longrope_mixed', 'longrope_dynamic', or 'none'. If rope_method is not 'none', the RoPE module
            will be replaced.
        max_position_embeddings (int, optional): Maximum number of position embeddings (after scaled).
        model_class (type, optional): The class of the model to load. Defaults to AutoModelForCausalLM.
        config (AutoConfig, optional): The configuration for the model. Defaults to None.
        rope_params (dict, optional): Additional parameters for the RoPE method. Defaults to None.
        attn_sliding_window (int, optional): The size of the attention sliding window (after scaled). Defaults to None.
        save_memory (bool, optional): Whether to save memory by replacing certain methods in the model. Defaults to False.
        **model_args: Additional keyword arguments to pass to the model.

    Returns:
        model: The loaded model.

    Raises:
        ValueError: If an unsupported RoPE method is specified.

    """
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    # NOTE: please force using attn_implementation="flash_attention_2" for now
    if hasattr(config, 'sliding_window') and config.sliding_window is not None:
        original_max_position_embeddings = config.sliding_window
        if attn_sliding_window > 0:
            logger.info(f"Change attention sliding window size: {config.sliding_window} => {attn_sliding_window}")
            config.sliding_window = attn_sliding_window
    else:
        original_max_position_embeddings = config.max_position_embeddings

    layer_num = config.num_hidden_layers
    head_size = config.hidden_size // config.num_attention_heads
    half_head_size = head_size // 2

    if max_position_embeddings is None:
        max_position_embeddings = config.max_position_embeddings
    config.max_position_embeddings = max_position_embeddings
    scaling_factor = max_position_embeddings / original_max_position_embeddings

    logger.info(f'[RoPE Method] {rope_method}')
    need_replace_rope = False
    if rope_method == 'pi':
        config.rope_scaling = {'type': 'linear', 'factor': scaling_factor}
    elif rope_method == 'dy_ntk':
        config.rope_scaling = {'type': 'dynamic', 'factor': scaling_factor}
    elif rope_method is not None and rope_method != 'none':
        need_replace_rope = True

    model = model_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        **model_args,
    )

    if save_memory:
        replace_methods(model)

    if need_replace_rope:
        if config.model_type == 'mistral' or config.model_type == 'mixtral':
            rope_model_type = 'mistral'
        else:
            if not (config.model_type == 'llama' or config.model_type == 'phi3'):
                logger.warning(f'Setting model type to llama for unrecognized model type: {config.model_type}')
            rope_model_type = 'llama'
        rope_class = None
        rope_args = {
            'original_max_position_embeddings': original_max_position_embeddings,
            'max_position_embeddings': max_position_embeddings,
            'scale': scaling_factor,
            'base': getattr(config, 'rope_embedding_base', getattr(config, 'rope_theta', None)),
            'model_type': rope_model_type,
        }
        if rope_method == 'yarn':
            rope_class = YaRNScaledRotaryEmbedding
        elif rope_method.startswith('longrope'):
            rescale_factors = np.loadtxt(open(rope_params['longrope_params_path'], 'rb'), delimiter=',', skiprows=0)
            if rescale_factors.shape == (half_head_size, ):
                rescale_factors = np.tile(rescale_factors.reshape((1, half_head_size)), (layer_num, 1))
            elif rescale_factors.shape != (layer_num, half_head_size):
                raise ValueError(f'misaligned shape for LongRoPE rescale factors: {rescale_factors.shape}')
            rope_args['rescale_factors'] = rescale_factors
            rope_args['magnitude_scaling_policy'] = rope_params['longrope_scaling_policy']
            if rope_method == 'longrope':
                rope_class = LongRoPEScaledRotaryEmbedding
            elif rope_method == 'longrope_mixed':
                rope_class = MixedLongRoPEScaledRotaryEmbedding
                # TODO: use hook to get the original embeddings
                original_rope = model.model.layers[0].self_attn.rotary_emb
                tmp_input = torch.zeros(size=(max_position_embeddings, ))
                original_embeddings = original_rope(tmp_input)
                rope_args['start_token_idx'] = rope_params['start_token_idx']
                rope_args['original_embeddings'] = original_embeddings
            elif rope_method == 'longrope_dynamic':
                rope_class = DynamicLongRoPEScaledRotaryEmbedding
        if rope_class is None:
            raise ValueError(f'Unsupported RoPE method: {rope_method}')
        logger.info(f'[RoPE Args]{rope_args}')
        return replace_rope(model, rope_class, rope_args)
    else:
        return model
