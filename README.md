# image-adversarial-noise

Perfoming white-box adversarial attacks on image classification models.

## Installation

This package was created and packaged using [Poetry](https://python-poetry.org/), and can be installed with `pip`:

```bash
git clone https://github.com/Acusick1/image-adversarial-noise.git
cd image-adversarial-noise
pip install .
```

Alternatively you can install directly from the `requirements.txt` as normal:

```bash
pip install requirements.txt
```

This is particularly for development purposes when poetry is not installed as it will also include the necessary dev dependencies.

## Generate adversarial image

To generate an adversarial image run the top -level [`gen_adversarial_image.py`](gen_adversarial_image.py) scipt, providing a path to an existing image, along with optional arguments such as `model_name` (default: resnet50) and `target_class` (default: random). A typical run command could look something like:

```bash
python gen_adversarial_image.py path/to/image.jpg --model_name resnet50.a1_in1k --target_class 123 --epsilon 0.05 --max_steps 20 --prob_threshold 0.5
```

Run `python gen_adversarial_image.py -h` for more input options and descriptions.

Note this script currently only supports image classification models that have been trained on the [imagenet-1k dataset](https://huggingface.co/datasets/imagenet-1k)!

## Exploration notebook

The package also contains an [exploration notebook](explore.ipynb) that was to develop and validate the final solution. Note that to run this notebook you should also install the dev dependencies contained in the [`pyproject.toml`](pyproject.toml) file.
