from __future__ import print_function

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


def main():
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--epochs", "-e", type=int, default=2)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="MSCOCO")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()


    if args.cuda and not torch.cuda.is_available():
        print("WARNING: torch.cuda not available, using CPU.")
        args.cuda = 0

    if args.cuda:
        kwargs = {'num_workers': 1, 'pin_memory': False}
    else:
        kwargs = {}

    print("=====================")
    print("TEST MODE")
    print("=====================")

    print("=====================")
    print("TORCH VERSION:", torch.__version__)
    print("BATCH SIZE:", args.batch_size)
    print("EPOCHS:", args.epochs)
    print("CUDA:", args.cuda)
    print("DATASET:", args.dataset)
    print("IMAGE SIZE:", args.image_size)
    print("=====================\n")

    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    for e in range(args.epochs):
        batch_id = 0
        for x in train_loader:
            if batch_id < 10 or batch_id % 500 == 0:
                print("Processing batch:", batch_id)
            batch_id += 1

    print("\nDone :)")


if __name__ == "__main__":
    main()
