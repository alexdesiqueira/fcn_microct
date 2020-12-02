# Large-scale fiber segmentation on unidirectional fiber beds using fully convolutional neural networks

## Alexandre Fioravante de Siqueira<sup>1,2,3</sup>, Daniela Mayumi Ushizima<sup>1,2</sup>, Stéfan J. van der Walt<sup>1,2</sup>

<sup>1</sup> _Berkeley Institute for Data Science, University of California, Berkeley, USA_

<sup>2</sup> _Lawrence Berkeley National Laboratory, Berkeley, USA_

<sup>3</sup> _Institute for Globally Distributed Open Research and Education (IGDORE)_

* This paper is available on: [[arXiv]](https://...)


## Downloading Larson et al's data

This study uses neural networks to process fibers in fiber beds, using Larson et al's data [1]. To be able to reproduce our study, it is necessary to download that data. For that, you will need a login at the [Globus](https://www.globus.org/) platform.

Larson et al's dataset is available at http://dx.doi.org/doi:10.18126/M2QM0Z. The samples we used are at the folder `data/Recons/Bunch2WoPR`.


## Preparing the PC to run the code locally

To download this repository to your machine, please use `git`. It can be downloaded freely [at the project's page](https://git-scm.com/downloads).

When `git` is installed, the following command on a [Linux](https://help.gnome.org/users/gnome-terminal/stable/)/[Mac OS](https://support.apple.com/guide/terminal/welcome/mac) Terminal or a [Windows PowerShell](https://docs.microsoft.com/en-us/powershell/scripting/windows-powershell/install/installing-windows-powershell?view=powershell-7) downloads this repository.

```bash
$ git clone https://github.com/alexdesiqueira/fcn_microct.git fcn_microct
```

The `$` represents the Terminal prompt. For more information on how to use `git`,  please check [its documentation](https://git-scm.com/doc).

You need Python installed to execute the code. We recommend using the [Anaconda distribution](https://www.anaconda.com/products/individual); all necessary tools are pre-installed. For installation instructions and packages to different operating systems, please refer to their [downloads page](https://www.anaconda.com/products/individual#Downloads). The following command installs the necessary dependencies:

```bash
$ pip install -r requirements.txt
```

The `$` represents the Terminal prompt. Now we are ready to use this repository.


## Training a neural network

After downloading Larson et al's original data and preparing the PC to run the code in this repository, you can use the script `train.py` to train the neural networks in the input data. For example, the command

```bash
$ python train.py -n 'tiramisu_3d' -t 'tiramisu-67' -w 'larson_tiramisu_3d-67.hdf5' -e 5 -b 2
```

will train a 3D Tiramisu-67 for 5 epochs, with a batch size of 2 using Larson et al's input data. The resulting weights will be stored at `larson_tiramisu_3d-67.hdf5`.

### Arguments

* `-n`, `--network` : convolutional network to be used in the training. **Available networks:** `'tiramisu'`, `'tiramisu_3d'`, `'unet'`, `'unet_3d'`.

* `-t`, `--tiramisu_model` : when the network used is a tiramisu, the model to be used. Not necessary when using U-Nets. **Available models:** `'tiramisu-56'`, `'tiramisu-67'`.

* `-v`, `--train_vars` : JSON file containing the training variables `'target_size'`, `'folder_train'`, `'folder_validate'`, `'training_images'`, `'validation_images'`. **Defaults:** based on `constants.py` to train using Larson et al samples.

An example of a JSON file follows:

```json
{
    "target_size": [64, 64, 64],
    "folder_train": "data/train",
    "folder_validate": "data/validate",
    "training_images": 1000,
    "validation_images": 200
}
```

* `-b`, `--batch_size` : size of the batches used in the training (optional). **Default:** 2.

* `-e`, `--epochs` : how many epochs are used in the training (optional). **Default:** 5.

* `-w`, `--weights` : output containing weight coefficients. **Default:** `weights_<NETWORK>.hdf5`.


## Predicting on Larson et al's data

After training one of the architectures into the input data — or if you would like to use one of weights we made available — you can use the script `predict.py` to predict results — i.e., use the network to separate regions of interest into your data. For example, the command

```bash
$ python predict.py -n 'tiramisu_3d' -t 'tiramisu-67' -w 'larson_tiramisu_3d-67.hdf5'
```

will separate fibers in Larson et al's input data using a 3D Tiramisu-67, with weights contained in the file `larson_tiramisu_3d-67.hdf5`.

### Arguments

* `-n`, `--network` : convolutional network to be used in the prediction. **Available networks:** `'tiramisu'`, `'tiramisu_3d'`, `'unet'`, `'unet_3d'`.

* `-t`, `--tiramisu_model` : when the network used is a tiramisu, the model to be used. Not necessary when using U-Nets. **Available models:** `'tiramisu-56'`, `'tiramisu-67'`.

* `-v`, `--train_vars` : JSON file containing the training variables `'folder'`, `'path'`, `'file_ext'`, `'has_goldstd'`, `'path_goldstd'`, `'segmentation_interval'`, `'registered_path'`. **Defaults:** based on `constants.py` to train using Larson et al samples.

An example of a JSON file follows:

```json
{
    "folder": "data",
    "path": "data/test",
    "file_ext": ".tif",
    "has_goldstd": true,
    "path_goldstd": "data/test/label",
    "segmentation_interval": null,
    "registered_path": null
}
```

* `-w`, `--weights` : file containing weight coefficients to be used on the prediction.


## HOW-TO: Reproducing our study

### Preparing the training samples

After downloading Larson et al's data, on the folder `fullconvnets`, start a Python prompt — e.g, Python interpreter, IPython, Jupyter Notebook. First, we import the library `prepare.py`:

```python
>>> import prepare
```

After importing `prepare`, we copy the training samples we will use, as defined in `constants.py`. Use the function `prepare.copy_training_samples()`:

```python
>>> prepare.copy_training_samples()
```

Then, we crop the images to fit the network input. If you would like to train the 2D networks, the following statement crops the training images and their labels:

```python
>>> prepare.crop_training_images()
```

To crop the training samples and their labels for the 3D networks, use the following statement:

```python
>>> prepare.crop_training_chunks()
```

### Training the networks

The following commands in a Terminal/PowerShell will train the four networks using the downloaded and prepared data, according to our study:

```bash
$ python train.py -n 'unet' -w 'larson_unet.hdf5' -e 5 -b 2
$ python train.py -n 'unet_3d' -w 'larson_unet_3d.hdf5' -e 5 -b 2
$ python train.py -n 'tiramisu' -t 'tiramisu-67' -w 'larson_tiramisu-67.hdf5' -e 5 -b 2
$ python train.py -n 'tiramisu_3d' -t 'tiramisu-67' -w 'larson_tiramisu_3d-67.hdf5' -e 5 -b 2
```

### Predicting using the trained networks

The following commands in a Terminal/PowerShell will predict the data using the four trained networks:

[TODO Continue here!]


## References

[[1]](https://www.sciencedirect.com/science/article/abs/pii/S1359835X18304603) Larson, N. M., Cuellar, C. & Zok, F. W. X-ray computed tomography of microstructure evolution during matrix impregnation and curing in unidirectional fiber beds. Composites Part A: Applied Science and Manufacturing 117, 243–259 (2019).
