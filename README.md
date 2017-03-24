# fast-neural-style :city_sunrise: :rocket:

Train model

```bash
python -u neural_style/neural_style.py train --cuda 1 --dataset MSCOCO/ --vgg-model vgg-model/ --save-model-dir saved-models/
```

Stylize image using saved model

```
python neural_style/neural_style.py eval --content-image </path/to/content/image> --output-image </path/to/output/image> --model </path/to/saved/model>
```
