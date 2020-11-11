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

* `-n`, `--network` : convolutional network to be used in the training. Available networks: `'tiramisu'`, `'tiramisu_3d'`, `'unet'`, `'unet_3d'`.

* `-t`, `--tiramisu_model` : when the network used is a tiramisu, the model to be used. Not necessary when using U-Nets. Available models: `'tiramisu-56'`, `'tiramisu-67'`.

* `-v`, `--train_vars` : JSON file containing the training variables `'target_size'`, `'folder_train'`, `'folder_validate'`, `'training_images'`, `'validation_images'`. Defaults: based on `constants.py` to train using Larson et al samples.

An example of a JSON file follows:

```json
{
    "target_size": [64, 64, 64],
    "folder_train": "tests/data/train",
    "folder_validate": "tests/data/validate",
    "training_images": 1536,
    "validation_images": 1024
}
```

* `-b`, `--batch_size` : size of the batches used in the training (optional). Default: 2.

* `-e`, `--epochs` : how many epochs are used in the training (optional). Default: 5.

* `-w`, `--weights` : output containing weight coefficients. Default: `weights_<NETWORK>.hdf5`.

## Predicting on Larson et al's data


## References

[1] Larson, N. M., Cuellar, C. & Zok, F. W. X-ray computed tomography of microstructure evolution during matrix impregnation and curing in unidirectional fiber beds. Composites Part A: Applied Science and Manufacturing 117, 243–259 (2019).
