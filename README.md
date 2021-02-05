# fast-neural-style :city_sunrise: :rocket:

**NOTICE**: This codebase is no longer maintained, please use the codebase from pytorch examples repository available at [pytorch/examples/fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style).

This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content of an image with the style of another image. For example, here is a photograph of a door arch rendered in the style of a stained glass painting.

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). The saved-models for examples shown in the README can be downloaded from [here](https://www.dropbox.com/s/gtwnyp9n49lqs7t/saved-models.zip?dl=0).

**DISCLAIMER**: This implementation is also a part of the [pytorch examples](https://github.com/pytorch/examples/tree/master/fast_neural_style) repository. Implementation in this repository uses pretrained Caffe2 VGG whereas the pytorch examples repository implementation uses pretrained Pytorch VGG. The two VGGs have different preprocessings which results in different `--content-weight` and `--style-weight` parameters. The styled output images also look slightly different.

<p align="center">
    <img src="images/style-images/mosaic.jpg" height="200px">
    <img src="images/content-images/amber.jpg" height="200px">
    <img src="images/output-images/amber-mosaic.jpg" height="440px">
</p>

## Requirements
The program is written in Python, and uses [pytorch](http://pytorch.org/), [scipy](https://www.scipy.org). A GPU is not necessary, but can provide a significant speed up especially for training a new model. Regular sized images can be styled on a laptop, desktop using saved models.

## Setup the environnment

### Run with virtualenv

Create a virtualenv with python3.5 or python3.6. Older versions are not supported due to a lack of compatibilty with pytorch.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run with Docker

Build the image:
```bash
docker build . -t fast-neural-style
```

Run the container:
```bash
docker run --rm --volume "$(pwd)/:/data" style eval --content-image /data/image.jpg --model /app/saved-models/mosaic.pth --output-image /data/output.jpg --cuda 0
```

## Usage
Stylize image
```
python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0
```
* `--content-image`: path to content image you want to stylize.
* `--model`: saved model to be used for stylizing the image (eg: `mosaic.pth`)
* `--output-image`: path for saving the output image.
* `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Train model
```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --vgg-model-dir </path/to/vgg/folder> --save-model-dir </path/to/save-model/folder> --epochs 2 --cuda 1
```

There are several command line arguments, the important ones are listed below
* `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. I used COCO 2014 Training images dataset [80K/13GB] [(download)](http://mscoco.org/dataset/#download).
* `--style-image`: path to style-image.
* `--vgg-model-dir`: path to folder where the vgg model will be downloaded.
* `--save-model-dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Refer to ``neural_style/neural_style.py`` for other command line arguments.

## Models

Models for the examples shown below can be downloaded from [here](https://www.dropbox.com/s/gtwnyp9n49lqs7t/saved-models.zip?dl=0) or by running the script ``download_styling_models.sh``.

<div align='center'>
  <img src='images/content-images/amber.jpg' height="174px">
</div>

<div align='center'>
  <img src='images/style-images/mosaic.jpg' height="174px">
  <img src='images/output-images/amber-mosaic.jpg' height="174px">
  <img src='images/output-images/amber-candy.jpg' height="174px">
  <img src='images/style-images/candy.jpg' height="174px">
  <br>
  <img src='images/style-images/starry-night-cropped.jpg' height="174px">
  <img src='images/output-images/amber-starry-night.jpg' height="174px">
  <img src='images/output-images/amber-udnie.jpg' height="174px">
  <img src='images/style-images/udnie.jpg' height="174px">
</div>
