import argparse
from pathlib import Path

import torch


def extract_state_dict(filename: str, save_path: str = None):
    assert filename.endswith('-full.pt')
    full_state = torch.load(filename)

    def remove_prefix(string: str, prefix: str) -> str:
        return string[len(prefix):] if isinstance(prefix, str) \
                                        and string[0:min(len(string), len(prefix))] == prefix else string

    def trunc_hf_dict_key(model_dict):
        new_dict = {}
        for key, val in model_dict.items():
            new_key = remove_prefix(key, 'hf_model.')
            new_dict[new_key] = val
        return new_dict

    model_state_dict = trunc_hf_dict_key(full_state['model'])
    if save_path is None:
        torch.save(model_state_dict, Path(filename).parent / 'pytorch_model.bin')
    else:
        assert not Path(save_path).is_file()
        Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(model_state_dict, Path(save_path) / 'pytorch_model.bin')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    extract_state_dict(args.filename)
