import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg16 import Vgg16


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda and not torch.cuda.is_available():
        print("WARNING: torch.cuda not available, using CPU.")
        args.cuda = 0

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    print("=====================")
    print("CURRENT TIME:", time.ctime())
    print("PYTHON VERSION:", sys.version)
    print("PYTORCH VERSION:", torch.__version__)
    print("BATCH SIZE:", args.batch_size)
    print("EPOCHS:", args.epochs)
    print("RANDOM SEED:", args.seed)
    print("CUDA:", args.cuda)
    print("LEARNING RATE:", args.lr)
    print("STYLE IMAGE:", args.style_image)
    print("CONTENT WEIGHT:", args.content_weight)
    print("STYLE WEIGHT:", args.style_weight)
    print("DATASET:", args.dataset)
    print("SAVE-MODEL DIRECTORY:", args.save_model_dir)
    print("STYLE SIZE:", args.style_size)
    print("=====================\n")

    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    transformer = TransformerNet()
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model, "vgg16.weight")))

    if args.cuda:
        transformer.cuda()
        vgg.cuda()

    style = utils.tensor_load_rgbimage(args.style_image, args.style_size)
    style = style.repeat(args.batch_size, 1, 1, 1)
    style = utils.preprocess_batch(style)
    if args.cuda:
        style = style.cuda()
    style_v = Variable(style, volatile=True)
    utils.subtract_imagenet_mean_batch(style_v)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))
            if args.cuda:
                x = x.cuda()

            y = transformer(x)

            xc = Variable(x.data.clone(), volatile=True)

            utils.subtract_imagenet_mean_batch(y)
            utils.subtract_imagenet_mean_batch(xc)

            features_y = vgg(y)
            features_xc = vgg(xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

    # save model
    transformer.eval()
    transformer.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer, save_model_path)

    print("\nDone :)")


def check_paths(args):
    flag_exit = 0
    if not os.path.exists(args.vgg_model):
        print("ERROR: Directory", "\"" + args.vgg_model + "\"", "doesn't exist. Please create the directory.")
        flag_exit = 1
    if not os.path.exists(args.save_model_dir):
        print("ERROR: Directory", "\"" + args.save_model_dir + "\"", "doesn't exist. Please create the directory.")
        flag_exit = 1
    if flag_exit:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--epochs", "-e", type=int, default=2)
    parser.add_argument("--vgg-model", "-m", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--style-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--style-image", type=str, default="images/style-images/wave.jpg")
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--save-model-dir", type=str, required=True)
    parser.add_argument("--train", type=int, default=1)

    args = parser.parse_args()
    check_paths(args)

    train(args)


if __name__ == "__main__":
    main()
