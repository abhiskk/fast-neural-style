import argparse

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from transformer_net import TransformerNet
from vgg16 import Vgg16
import utils
import os
import sys


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 4, 'pin_memory': False}
    else:
        kwargs = {}

    print("=====================")
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
    print("CHECKPOINT DIR:", args.checkpoint_dir)
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
            # pass images through the TransformerNet
            y = transformer(x)
            features_y = vgg(y)
            xc = Variable(x.data.clone(), volatile=True)
            features_xc = vgg(xc)
            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch,:,:])

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]

            if (batch_id + 1) % args.log_interval == 0:

                # TODO: Save some stylized images from the training set

                mesg = "Epoch {}:\t[{}/{}]\tcontent:{:.2f}\tstyle:{:.2f}".format(
                    e + 1, count, len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1)
                )
                print(mesg)
        with open(args.checkpoint_dir + "/epoch_" + str(e + 1) + ".model", "w") as file_pointer:
            torch.save(transformer, file_pointer)

    print("\nDone :)")


def stylize(args):
    print("=====================")
    print("PYTHON VERSION:", sys.version)
    print("PYTORCH VERSION:", torch.__version__)
    print("CUDA:", args.cuda)
    print("SAVED MODEL PATH:", args.saved_model_path)
    print("CONTENT IMAGE:", args.content_image)
    print("IMAGE SIZE:", args.image_size)
    print("SAVE IMAGE PATH:", args.save_image_path)
    print("=====================\n")

    model = torch.load(args.saved_model_path)
    model.eval()
    print("-" * 50)
    print(model)
    print("-" * 50 + "\n")

    content = utils.tensor_load_rgbimage(args.content_image, args.image_size)
    content = content.unsqueeze(0)
    content = utils.preprocess_batch(content)
    if args.cuda:
        content = content.cuda()
    content = Variable(content)
    stylized_content = model(content)
    if args.cuda:
        stylized_content = stylized_content.cpu()

    utils.deprocess_img_and_save(stylized_content.data.numpy(), args.save_image_path)
    print("Styled image saved at:", args.save_image_path)


def main():
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--epochs", "-e", type=int, default=2)
    parser.add_argument("--vgg-model", "-m", type=str, default="vgg-model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="MSCOCO")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--style-size", default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg")
    parser.add_argument("--content-weight", type=float, default=1.)
    parser.add_argument("--style-weight", type=float, default=5.)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--saved-model-path", type=str, default=None)
    parser.add_argument("--content-image", type=str, default=None)
    parser.add_argument("--save-image-path", type=str, default=None)
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("WARNING: torch.cuda not available, using CPU.")
        args.cuda = 0

    if args.train:
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
