# fast-neural-style :city_sunrise: :rocket:

## Requirements
The program is written in Python, and uses [PyTorch](http://pytorch.org/), [Scipy](https://www.scipy.org). A GPU is not necessary, but can provide a significant speed up.

## Usage
Stylize image
```
python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image>
```
* `--content-image`: path to content image you want to stylize.
* `--model`: saved model to be used for stylizing the image (eg: `mosaic.model` present under `saved-models/`)
* `--output-image`: path for saving the output image.
* `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will half the height and width of content-image)

Train model
```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --vgg </path/to/vgg/folder> --save-model-dir </path/to/save-models/folder> --epochs 2 --cuda 1
```

There are several command line arguments, the important ones are listed below
* `--dataset`: path to training dataset, I used COCO 2014 Training images dataset [80K/13GB] [[download]](http://mscoco.org/dataset/#download).
* `--vgg`: path to folder where the vgg model will be downloaded.
* `--save-model-dir`: path to folder where trained model will be saved
* `--epochs`: train for these many iterations. The default is 2.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Refer to ``neural_style/neural_style.py`` for other command line arguments.
