import os
import json
import glob
import shutil
import logging
import argparse

import numpy as np


logger = logging.getLogger(__file__)
template_folder = os.path.join(os.path.split(__file__)[0], "checkpoint_template")


def check_transformers_version(version: str):
    if version is None:
        from transformers import __version__ as version
    upper_version = '.'.join(version.split('.')[:2])
    assert os.path.isdir(os.path.join(template_folder, upper_version)), \
        f'Unsupported transformers version: {version}'
    return upper_version


def main(
    base_model_path: str,
    weights_path: str,
    long_factor_path: str,
    short_factor_path: str,
    output_path: str,
    max_position_embeddings: int,
    sliding_window: int = None,
):
    transformers_version = check_transformers_version(args.transformers_version)
    template_path = os.path.join(template_folder, transformers_version)
    with open(os.path.join(base_model_path, "config.json")) as f:
        config = json.loads(f.read())
    model_type: str = config["model_type"]
    assert model_type in ["llama", "mistral"], \
        f"Unsupported base model type: {model_type}"
    logger.info(f"Copy configurations and tokenizer from base model: {base_model_path}")
    os.makedirs(output_path)
    for src in glob.glob(os.path.join(base_model_path, "*")):
        if not (("pytorch_model" in src) or ("safetensors" in src)):
            shutil.copy(src, output_path)
    logger.info(f"Rewrite config.json with {long_factor_path} and {short_factor_path}")
    long_factor = np.loadtxt(long_factor_path, delimiter=",", skiprows=0).tolist()
    short_factor = np.loadtxt(short_factor_path, delimiter=",", skiprows=0).tolist()
    config["model_type"] = "phi_longrope"
    config["architectures"] = ["PhiLongRoPEForCausalLM"]
    config["auto_map"] = {
      "AutoConfig": "configuration_phi_longrope.PhiLongRoPEConfig",
      "AutoModelForCausalLM": "modeling_phi_longrope.PhiLongRoPEForCausalLM"
    }
    config["rope_scaling"] = {
        "type": "longrope",
        "long_factor": long_factor,
        "short_factor": short_factor,
    }
    config["original_max_position_embeddings"] = config["max_position_embeddings"]
    config["max_position_embeddings"] = max_position_embeddings
    if sliding_window is not None:
        config["sliding_window"] = sliding_window
    with open(os.path.join(output_path, "config.json"), "w") as f:
        f.write(json.dumps(config, indent=2))
    logger.info(f"Add remote code")
    for template in glob.glob(os.path.join(template_path, "*")):
        with open(template, encoding="utf8") as f:
            lines = f.read().split(os.linesep)
        filtered_lines = []
        flag = True
        for line in lines:
            if "COMPONENTS BEGIN" in line:
                if model_type.upper() not in line:
                    flag = False
            elif "COMPONENTS END" in line:
                flag = True
            elif flag:
                filtered_lines.append(line)
        with open(template.replace(template_path, output_path), "w", encoding="utf8") as f:
            f.write(os.linesep.join(filtered_lines))
    logger.info(f"Copy model weights: {weights_path}")
    shutil.copy(weights_path, output_path)
    logger.info(f"Finished. Output: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%m-%d %H:%M",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--long-factor", type=str)
    parser.add_argument("--short-factor", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--max-position-embeddings", type=int)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--transformers-version", type=str, default=None)
    args = parser.parse_args()
    main(
        base_model_path=args.base_model,
        weights_path=args.weights,
        long_factor_path=args.long_factor,
        short_factor_path=args.short_factor,
        output_path=args.output,
        max_position_embeddings=args.max_position_embeddings,
        sliding_window=args.sliding_window,
    )
