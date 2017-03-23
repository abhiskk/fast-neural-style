from transformer_net import TransformerNet
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-model", type=str, required=True)
    parser.add_argument("--cpu-model", type=str, required=True)
    args = parser.parse_args()

    tr = torch.load(args.gpu_model)
    tr.eval()
    tr.cpu()

    torch.save(tr, args.cpu_model)


if __name__ == "__main__":
    main()