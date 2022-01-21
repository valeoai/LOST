# disclaimer: inspired by MoCo and PyContrast official repos.

import pickle as pkl
import torch
import argparse


def _load_pytorch_weights(file_path):
    checkpoint = torch.load(file_path, map_location="cpu")
    if "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
    elif "network" in checkpoint:
        weights = checkpoint["network"]
    else:
        for key in list(checkpoint.keys()):
            if key.startswith('module.'):
                # remove prefix
                checkpoint[key[len('module.'):]] = checkpoint[key].cpu()
                del checkpoint[key]
        weights = checkpoint
    return weights


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert Models')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to PyTorch RN-50 model')
    parser.add_argument('--output', type=str, default=None,
                        help='Destination path')
    args = parser.parse_args()

    state_dict = _load_pytorch_weights(args.input)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("fc."):
            print(f"Skip fully connected params {k}")
            continue
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        k = k.replace("layer1", "res2")
        k = k.replace("layer2", "res3")
        k = k.replace("layer3", "res4")
        k = k.replace("layer4", "res5")
        k = k.replace("bn1", "conv1.norm")
        k = k.replace("bn2", "conv2.norm")
        k = k.replace("bn3", "conv3.norm")
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")

        k2 = old_k
        k2 = k2.replace(".downsample.1.", ".branch1_bn.")
        k2 = k2.replace(".downsample.1.", ".branch1_bn.")
        k2 = k2.replace(".downsample.0.", ".branch1.")
        k2 = k2.replace(".conv1.", ".branch2a.")
        k2 = k2.replace(".bn1.", ".branch2a_bn.")
        k2 = k2.replace(".conv2.", ".branch2b.")
        k2 = k2.replace(".bn2.", ".branch2b_bn.")
        k2 = k2.replace(".conv3.", ".branch2c.")
        k2 = k2.replace(".bn3.", ".branch2c_bn.")
        k2 = k2.replace("layer1.", "res2.")
        k2 = k2.replace("layer2.", "res3.")
        k2 = k2.replace("layer3.", "res4.")
        k2 = k2.replace("layer4.", "res5.")
        print(f"{old_k} -> {k} vs {k2}")

        new_state_dict[k] = v.numpy()

    res = {"model": new_state_dict,
           "__author__": "MoCo",
           "matching_heuristics": True}

    with open(args.output, "wb") as f:
        pkl.dump(res, f)
