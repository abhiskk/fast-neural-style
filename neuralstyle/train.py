from __future__ import print_function

import argparse

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from transformernet import TransformerNet
from vgg16 import Vgg16
import utils
import os


def main():
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--epochs", "-e", type=int, default=2)
    parser.add_argument("--model", "-m", type=str, default="model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="MSCOCO")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--style-image", type=str, default="style-images/mosaic.jpg")
    parser.add_argument("--content-weight", type=float, default=8.)
    parser.add_argument("--style-weight", type=float, default=5e-4)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("WARNING: torch.cuda not available, using CPU.")
        args.cuda = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    print("=====================")
    print("BATCH SIZE:", args.batch_size)
    print("EPOCHS:", args.epochs)
    print("RANDOM SEED:", args.seed)
    print("CUDA:", args.cuda)
    print("LEARNING RATE:", args.lr)
    print("STYLE IMAGE:", args.style_image)
    print("CONTENT WEIGHT:", args.content_weight)
    print("STYLE WEIGHT:", args.style_weight)
    print("TV WEIGHT:", args.tv_weight)
    print("=====================\n")

    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    # TODO: Add shuffling of data
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    transformer = TransformerNet()
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16()
    utils.init_vgg16(args.model)
    vgg.load_state_dict(torch.load(os.path.join(args.model, "vgg16.weight")))

    if args.cuda:
        transformer.cuda()
        vgg.cuda()

    style = utils.tensor_load_rgbimage(args.style_image, args.image_size)
    style = style.repeat(args.batch_size, 1, 1, 1)
    style = utils.preprocess_batch(style)
    if args.cuda:
        style = style.cuda()
    style_v = Variable(style)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_tv_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))
            if args.cuda:
                x = x.cuda()
            # create a copy of original images for calculating content loss
            xc = x.clone()
            # pass images through the TransformerNet
            y = transformer(x)
            features_y = vgg(y)
            features_xc = vgg(xc)
            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch,:,:])

            tv_loss = args.tv_weight * ((torch.sum(torch.abs(y[:,:,1:,:] - y[:,:,:-1,:])) + torch.sum(torch.abs(y[:,:,:,1:] - y[:,:,:,:-1]))) / float(n_batch))

            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]
            agg_tv_loss += tv_loss.data[0]

            if (batch_id + 1) % 500 == 0:
                mesg = "Epoch {}:\t[{}/{}]\tcontent:{:.2f}\tstyle:{:.2f}\ttv:{:.2f}".format(
                    e, count, 80000,
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    agg_tv_loss / (batch_id + 1))
                print(mesg)

    print("\nDone :)")


if __name__ == "__main__":
    main()