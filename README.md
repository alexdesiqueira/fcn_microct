# A reusable pipeline for large-scale fiber segmentation on unidirectional fiber beds using fully convolutional neural networks

## Alexandre Fioravante de Siqueira<sup>1,2,3</sup>, Daniela Mayumi Ushizima<sup>1,2</sup>, Stéfan J. van der Walt<sup>1,2</sup>

<sup>1</sup> _Berkeley Institute for Data Science, University of California, Berkeley, USA_

<sup>2</sup> _Lawrence Berkeley National Laboratory, Berkeley, USA_

<sup>3</sup> _Institute for Globally Distributed Open Research and Education (IGDORE)_

* This paper is available on: [[arXiv]](https://...)


## Downloading Larson et al's data

This study uses neural networks to process fibers in fiber beds, using Larson et al's data [1]. To be able to reproduce our study, it is necessary to download that data. For that, you will need a login at the [Globus](https://www.globus.org/) platform.

Larson et al's dataset is available at http://dx.doi.org/doi:10.18126/M2QM0Z. We used twelve different datasets in total. We keep the same file identifiers Larson et al. used in their study, for fast cross-reference:

* **"232p1":** 
    * wet: folder `data/Recons/Bunch2WoPR/rec20160324_055424_232p1_wet_1cm_cont_4097im_1500ms_17keV_13_a.h5`
* **"232p3":**
    * wet: folder `data/Recons/Bunch2WoPR/rec20160318_191511_232p3_2cm_cont__4097im_1500ms_ML17keV_6.h5`
    * cured: folder `data/Recons/Bunch2WoPR/rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5`
    * cured registered: folder `data/Seg/Bunch2/rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5/Registered/Bunch2WoPR`
* **"235p1":**
    * wet: folder `data/Recons/Bunch2WoPR/rec20160324_123639_235p1_wet_0p7cm_cont_4097im_1500ms_17keV_14.h5`
* **"235p4":**
    * wet: folder `data/Recons/Bunch2WoPR/rec20160326_175540_235p4_wet_1p15cm_cont_4097im_1500ex_17keV_20.h5`
    * cured: folder `data/Recons/Bunch2WoPR/rec20160327_003824_235p4_cured_1p15cm_cont_4097im_1500ex_17keV_22.h5`
    * cured registered: folder `data/Seg/Bunch2/rec20160327_003824_235p4_cured_1p15cm_cont_4097im_1500ex_17keV_22.h5/Registered/Bunch2WoPR`
* **"244p1":**
    * wet: folder `data/Recons/Bunch2WoPR/rec20160318_223946_244p1_1p5cm_cont__4097im_1500ms_ML17keV_7.h5`
    * cured: folder `data/Recons/Bunch2WoPR/rec20160320_160251_244p1_1p5cm_cont_4097im_1500ms_ML17keV_9.h5`
    * cured registered: folder `data/Seg/Bunch2/rec20160320_160251_244p1_1p5cm_cont_4097im_1500ms_ML17keV_9.h5/Registered/Bunch2WoPR`
* **"245p1":** 
    * wet: folder `rec20160327_160624_245p1_wet_1cm_cont_4097im_1500ex_17keV_23.h5`

The first three numeric characters correspond to a material sample, and the last character correspond to different extrinsic factors, e.g. deformation. Despite being samples from similar materials, the reconstructed files presented several differences: different amount of ringing artifacts, intensity variation, noise, etc.

A copy of the folder structure is given at the Appendix, at the end of this file.


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

The following instructions can be used to reproduce the results from our manuscript. All CNN algorithms were implemented using [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) on a computer with two Intel Xeon Gold processors 6134 and two Nvidia GeForce RTX 2080 graphical processing units. Each GPU has 10 GB of RAM.

### Preparing the training samples

After downloading Larson et al's data, on the folder `fullconvnets`, start a Python prompt — e.g, [Python interpreter](https://docs.python.org/3.8/tutorial/interpreter.html), [IPython](http://ipython.org/), [Jupyter Notebook](https://jupyter.org/). First, we import the library `prepare.py`:

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

The following commands in a [Linux](https://help.gnome.org/users/gnome-terminal/stable/)/[Mac OS](https://support.apple.com/guide/terminal/welcome/mac) Terminal or a [Windows PowerShell](https://docs.microsoft.com/en-us/powershell/scripting/windows-powershell/install/installing-windows-powershell?view=powershell-7) will train the four networks using the downloaded and prepared data, according to our study:

```bash
$ python train.py -n 'unet' -w 'larson_unet.hdf5' -e 5 -b 2
$ python train.py -n 'unet_3d' -w 'larson_unet_3d.hdf5' -e 5 -b 2
$ python train.py -n 'tiramisu' -t 'tiramisu-67' -w 'larson_tiramisu-67.hdf5' -e 5 -b 2
$ python train.py -n 'tiramisu_3d' -t 'tiramisu-67' -w 'larson_tiramisu_3d-67.hdf5' -e 5 -b 2
```

### Predicting using the trained networks

The following commands in a [Linux](https://help.gnome.org/users/gnome-terminal/stable/)/[Mac OS](https://support.apple.com/guide/terminal/welcome/mac) Terminal or a [Windows PowerShell](https://docs.microsoft.com/en-us/powershell/scripting/windows-powershell/install/installing-windows-powershell?view=powershell-7) will predict the data using the four trained networks:

[TODO Check these commands]

```bash
$ python predict.py -n 'unet' -w 'larson_unet.hdf5'
$ python predict.py -n 'unet_3d' -w 'larson_unet_3d.hdf5'
$ python predict.py -n 'tiramisu' -t 'tiramisu-67' -w 'larson_tiramisu-67.hdf5'
$ python predict.py -n 'tiramisu_3d' -t 'tiramisu-67' -w 'larson_tiramisu_3d-67.hdf5'
```

## References

[[1]](https://www.sciencedirect.com/science/article/abs/pii/S1359835X18304603) Larson, N. M., Cuellar, C. & Zok, F. W. X-ray computed tomography of microstructure evolution during matrix impregnation and curing in unidirectional fiber beds. Composites Part A: Applied Science and Manufacturing 117, 243–259 (2019).


# Appendices

## Structure on Larson et al's data

This is the structure of Larson et al's folders we used in this study, for reference.

```bash
data/Recons/Bunch2WoPR/
├── rec20160318_191511_232p3_2cm_cont__4097im_1500ms_ML17keV_6.h5
│   ├── rec_SFRR_2600_B0p2_00159.tiff
│   ├── rec_SFRR_2600_B0p2_00160.tiff
│   ├── (...)
│   └── rec_SFRR_2600_B0p2_01158.tiff
├── rec20160318_223946_244p1_1p5cm_cont__4097im_1500ms_ML17keV_7.h5
│   ├── rec_SFRR_2600_B0p2_00000.tiff
│   ├── rec_SFRR_2600_B0p2_00001.tiff
│   ├── (...)
│   └── rec_SFRR_2600_B0p2_02159.tiff
├── rec20160320_160251_244p1_1p5cm_cont_4097im_1500ms_ML17keV_9.h5
│   ├── rec_SFRR_2600_B0p2_00000.tiff
│   ├── rec_SFRR_2600_B0p2_00001.tiff
│   ├── (...)
│   └── rec_SFRR_2600_B0p2_02159.tiff
├── rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5
│   ├── rec_SFRR_2600_B0p2_00000.tiff
│   ├── rec_SFRR_2600_B0p2_00001.tiff
│   ├── (...)
│   └── rec_SFRR_2600_B0p2_02159.tiff
├── rec20160324_055424_232p1_wet_1cm_cont_4097im_1500ms_17keV_13_a.h5
│   ├── rec_SFRR_2600_B0p2_00000.tiff
│   ├── rec_SFRR_2600_B0p2_00001.tiff
│   ├── (...)
│   └── rec_SFRR_2600_B0p2_02159.tiff
├── rec20160324_123639_235p1_wet_0p7cm_cont_4097im_1500ms_17keV_14.h5
│   ├── rec_SFRR_2600_B0p2_00000.tiff
│   ├── rec_SFRR_2600_B0p2_00001.tiff
│   ├── (...)
│   └── rec_SFRR_2600_B0p2_02159.tiff
├── rec20160326_175540_235p4_wet_1p15cm_cont_4097im_1500ex_17keV_20.h5
│   ├── rec_SFRR_2600_B0p2_00000.tiff
│   ├── rec_SFRR_2600_B0p2_00001.tiff
│   ├── (...)
│   └── rec_SFRR_2600_B0p2_02159.tiff
├── rec20160327_003824_235p4_cured_1p15cm_cont_4097im_1500ex_17keV_22.h5
│   ├── rec_SFRR_2600_B0p2_00000.tiff
│   ├── rec_SFRR_2600_B0p2_00001.tiff
│   ├── (...)
│   └── rec_SFRR_2600_B0p2_02159.tiff
└── rec20160327_160624_245p1_wet_1cm_cont_4097im_1500ex_17keV_23.h5
    ├── rec_SFRR_2600_B0p2_00000.tiff
    ├── rec_SFRR_2600_B0p2_00001.tiff
    ├── (...)
    └── rec_SFRR_2600_B0p2_02159.tiff

data/Seg/Bunch2/
├── rec20160320_160251_244p1_1p5cm_cont_4097im_1500ms_ML17keV_9.h5
│   └── Registered
│       └── Bunch2WoPR
│           ├── Reg_0001.tif
│           ├── Reg_0002.tif
│           ├── (...)
│           └──
├── rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5
│   └── Registered
│       └── Bunch2WoPR
│           ├── Reg_0001.tif
│           ├── Reg_0002.tif
│           ├── (...)
│           └── Reg_2160.tif
└── rec20160327_003824_235p4_cured_1p15cm_cont_4097im_1500ex_17keV_22.h5
    └── Registered
        └── Bunch2WoPR
            ├── Reg_0001.tif
            ├── Reg_0002.tif
            ├── (...)
            └── Reg_2160.tif
```