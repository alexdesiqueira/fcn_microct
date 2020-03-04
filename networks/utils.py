from models.unet import unet, unet_3d, unet_3d_3
from models.tiramisu import tiramisu, tiramisu_3d
from skimage import color, exposure, io, util
from tensorflow.keras.backend import clear_session

import constants as const
import csv
import evaluation
import numpy as np
import os


def folders_to_process(folder_base, folder_pred, subfolder_pred,
                       folder_gold_std, subfolder_gold_std):
    """Returns lists of folders with correspondents in both prediction and gold
    standard samples.

    Notes
    -----
       The input arguments generated from Larson's gold standard and our
    results are:

    FOLDER_BASE = '~/data/larson_2019/'
    FOLDER_PRED = 'data/<network>/'
    SUBFOLDER_PRED = 'predict'
    FOLDER_GOLD_STD = 'Seg/Bunch2/'
    SUBFOLDER_GOLD_STD = '19_Gray_image'

    Please replace <network> in FOLDER_PRED with 'unet', 'tiramisu', 'unet_3d',
    or 'tiramisu_3d', depending on the predictions you want to use.
    """
    pred_folders, gold_folders = [], []

    base_pred = os.path.join(folder_base, folder_pred)
    base_gold = os.path.join(folder_base, folder_gold_std)

    for folder in [base_pred, base_gold]:
        _assert_folder_exists(folder)

    # checking the path for prediction samples.
    pred_folders = _folder_samples(base_ref=base_pred,
                                   subfolder_ref=subfolder_pred)

    # now, checking the path for Larson's gold standard.
    gold_folders = _folder_samples(base_ref=base_gold,
                                   subfolder_ref=subfolder_gold_std)

    # ensuring the same samples are available in pred_folders and gold_folders
    pred_valid, gold_valid = _suitable_samples(pred_folders, gold_folders)

    return pred_valid, gold_valid


def imread_prediction(image):
    """Auxiliary function intended to be used with skimage.io.ImageCollection.
    Returns a binary prediction image — True when image > 0.5, False
    when image <= 0.5.
    """
    return util.img_as_float(io.imread(image)) > 0.5


def imread_goldstd(image):
    """Auxiliary function intended to be used with skimage.io.ImageCollection.
    Helps to read and process Larson et al's gold standard images.
    """
    return process_goldstd_images(io.imread(image))


def measure_all_coefficients(data_test, data_gt,
                             calc_coef={'matthews': True, 'dice': True},
                             save_coef=True,
                             filename='coefficients.csv'):
    """Measures all comparison coefficients between two input data.

    Example
    -------
    >>> from skimage import io
    >>> data_bin = io.ImageCollection(load_pattern='res_figures/binary/Test_TIRR_0_1p5_B0p2_*_bin.png',
                                      plugin=None)[1000:2000]
    >>> data_gt = io.ImageCollection(load_pattern='gt_figures/19_Gray_*.tif',
                                     plugin=None)
    >>> all_matthews, all_dice = measure_all_coefficients(data_bin,
                                                          data_gt,
                                                          save_coef=True)
    """
    _assert_same_length(data_test, data_gt)

    all_matthews, all_dice = ['matthews'], ['dice']

    for idx, img_test in enumerate(data_test):
        aux_gt = process_goldstd_images(data_gt[idx])

        _assert_compatible(img_test, aux_gt)

        aux_confusion = evaluation.confusion_matrix(aux_gt, img_test)
        if calc_coef['matthews']:
            all_matthews.append(evaluation.measure_matthews(aux_confusion))
        if calc_coef['dice']:
            all_dice.append(evaluation.measure_dice(aux_confusion))

    if save_coef:
        if calc_coef['matthews']:
            with open(filename, 'a+') as file_coef:
                coef_writer = csv.writer(file_coef, delimiter=',')
                coef_writer.writerow(all_matthews)
        if calc_coef['dice']:
            with open(filename, 'a+') as file_coef:
                coef_writer = csv.writer(file_coef, delimiter=',')
                coef_writer.writerow(all_dice)

    return all_matthews, all_dice


def montage_3d(array_input, grid_shape=(1, 80, 80)):
    """Create a montage of several single- or multichannel cubes.
    """
    ntiles_plane, ntiles_row, ntiles_col = [int(s) for s in grid_shape]

    _, n_planes, n_rows, n_cols = array_input.shape
    array_out = np.empty((n_planes * ntiles_plane,
                          n_rows * ntiles_row,
                          n_cols * ntiles_col),
                         dtype=array_input.dtype)

    slices_plane = [slice(n_planes * n,
                          n_planes * n + n_planes)
                    for n in range(ntiles_plane)]
    slices_row = [slice(n_rows * n,
                        n_rows * n + n_rows)
                  for n in range(ntiles_row)]
    slices_col = [slice(n_cols * n,
                        n_cols * n + n_cols)
                  for n in range(ntiles_col)]

    for idx_image, image in enumerate(array_input):
        idx_sp = idx_image % ntiles_plane
        idx_sr = idx_image // ntiles_col
        idx_sc = idx_image % ntiles_col
        array_out[slices_plane[idx_sp], slices_row[idx_sr], slices_col[idx_sc]] = image

    return array_out


def network_models(network='unet', window_shape=(288, 288), n_class=1,
                   preset_model='tiramisu-67'):
    """
    """
    if network in const.AVAILABLE_2D_NETS:
        model = _available_2d_nets(network,
                                   window_shape,
                                   n_class,
                                   preset_model)
    elif network in const.AVAILABLE_3D_NETS:
        model = _available_3d_nets(network,
                                   window_shape,
                                   n_class,
                                   preset_model)
    return model


def overlap_predictions(image, prediction):
    """Overlaps the prediction on top of the input image.

    Parameters
    ----------
    image : (M, N) array
        Input image.
    prediction : (M, N, P) array
        Predicted results.

    Returns
    -------
    output : (M, N) array
        Overlapped image.

    Notes
    -----
    P in prediction indicates the number of classes.
    """
    _assert_compatible(image, prediction)

    overlap = util.img_as_ubyte(color.gray2rgb(image))
    overlap[..., 0] = util.img_as_ubyte(prediction > 0.5)

    return overlap


def predict_on_chunk(data, weights, network='unet_3d', n_class=1, pad_width=16,
                     window_shape=(32, 32, 32), step=32):
    """
    """
    model = network_models(network,
                           window_shape=window_shape,
                           n_class=n_class)
    if network is None:
        raise(f'Network {network} not available.')
    model.load_weights(weights)

    data = np.pad(data, pad_width=pad_width)
    data_crop = np.vstack(np.hstack(
        util.view_as_windows(data,
                             window_shape=window_shape,
                             step=step)
    ))
    crop_steps = data_crop.shape[0]
    chunk_gen = tensor_generator(data_crop)

    results = model.predict(chunk_gen, steps=crop_steps, verbose=1)
    prediction = _aux_predict(results,
                              grid_shape=(1, 80, 80))

    return prediction


def predict_on_image(image, weights, network='unet', n_class=1, pad_width=16,
                     window_shape=(288, 288), step=256):
    """
    """
    model = network_models(network,
                           window_shape=window_shape,
                           n_class=n_class)
    if network is None:
        raise(f'Network {network} not available.')
    model.load_weights(weights)

    image = np.pad(image, pad_width=pad_width)
    image_crop = np.vstack(util.view_as_windows(image,
                                                window_shape=window_shape,
                                                step=step))
    crop_steps = image_crop.shape[0]
    image_gen = tensor_generator(image_crop)

    results = model.predict(image_gen, steps=crop_steps, verbose=1)
    prediction = _aux_predict(results)

    return prediction


def prediction_folder(network='unet'):
    """Returns the saving folder according to the network.
    """
    available_folders = {
        'tiramisu': const.FOLDER_PRED_TIRAMISU,
        'tiramisu_3d': const.FOLDER_PRED_TIRAMISU_3D,
        'unet': const.FOLDER_PRED_UNET,
        'unet_3d': const.FOLDER_PRED_UNET_3D,
        'unet_3d_3': const.FOLDER_PRED_UNET_3D_3LAYERS,
    }
    return available_folders.get(network, None)


def process_goldstd_images(data_goldstd):
    """
    """
    if np.unique(data_goldstd).size > 2:
        data_goldstd = data_goldstd == 217

    return util.img_as_bool(data_goldstd)


def process_sample(folder, data, weights, network='unet'):
    """Process the sample and overlaps the original
    image with the results.
    """
    FOLDER_PRED = os.path.join(folder, const.SUBFOLDER_PRED)
    FOLDER_OVER = os.path.join(folder, const.SUBFOLDER_OVER)

    for aux in [FOLDER_PRED, FOLDER_OVER]:
        if not os.path.isdir(aux):
            os.makedirs(aux)

    if network in const.AVAILABLE_3D_NETS:
        last_original_plane = None
        data = data.concatenate()

        n_planes, n_rows, n_cols = data.shape
        # data should be divisible by const.STEP_3D.
        if n_rows > n_planes:
            last_original_plane = n_planes
            data_complete = np.zeros((n_rows - n_planes, n_rows, n_cols),
                                     dtype='bool')
            # filling data if necessary, to ensure np.split works.
            data = np.concatenate((data, data_complete))
            n_planes, n_rows, n_cols = data.shape

        sections = n_planes / const.STEP_3D
        data = np.split(data,
                        indices_or_sections=sections)
        for idc, chunk in enumerate(data):
            # generating a list of possible filenames...
            filenames = []
            for num in range(idx*const.STEP_3D):
                filenames.append('%06d.png' % num)

            # ... then checking if the files exist, before processing a chunk.
            all_files_exist = all(
                [os.path.isfile(os.path.join(FOLDER_PRED, filename) for filename in filenames]
                )
            if not all_files_exist:
                prediction = predict_on_chunk(chunk,
                                              weights=weights,
                                              network=network,
                                              pad_width=const.PAD_WIDTH_3D,
                                              window_shape=const.WINDOW_SHAPE_3D)
                for idx, plane in enumerate(prediction):
                    current_plane = idx + idc*const.STEP_3D
                    # avoiding to save auxiliary slices with no info.
                    if last_original_plane and last_original_plane > current_plane:
                        filename = '%06d.png' % (current_plane)

                        io.imsave(os.path.join(FOLDER_PRED, filename),
                                  util.img_as_ubyte(plane))
                        overlap = overlap_predictions(chunk[idx], plane)
                        io.imsave(os.path.join(FOLDER_OVER, filename),
                                  util.img_as_ubyte(overlap))
            clear_session()  # resetting TensorFlow session state.

    elif network in const.AVAILABLE_2D_NETS:
        for idx, image in enumerate(data):
            filename = '%06d.png' % (idx)
            if not os.path.isfile(os.path.join(FOLDER_PRED, filename)):
                prediction = predict_on_image(image,
                                              weights=weights,
                                              network=network,
                                              pad_width=const.PAD_WIDTH,
                                              window_shape=const.WINDOW_SHAPE)
                io.imsave(os.path.join(FOLDER_PRED, filename),
                          util.img_as_ubyte(prediction))

            # if file doesn't exist, overlaps the results on the original.
            if not os.path.isfile(os.path.join(FOLDER_OVER, filename)):
                overlap = overlap_predictions(image, prediction)
                io.imsave(os.path.join(FOLDER_OVER, filename),
                          util.img_as_ubyte(overlap))

            clear_session()  # resetting TensorFlow session state.
    return None


def read_csv_coefficients(filename):
    """Reads csv coefficients saved in a file."""
    coefs = []
    csv_file = csv.reader(open(filename, 'r'))
    for row in csv_file:
        coefs.append(row)

    coefs_matthews, coefs_dice = coefs
    matthews = np.asarray(coefs_matthews[1:], dtype='float')
    dice = np.asarray(coefs_dice[1:], dtype='float')

    return matthews, dice


def regroup_image(image_set, grid_shape=None, pad_width=32,
                  multichannel=False):
    """
    """
    if multichannel:
        image_set = image_set[:,
                              pad_width:-pad_width,
                              pad_width:-pad_width,
                              :]
    else:
        image_set = image_set[:,
                              pad_width:-pad_width,
                              pad_width:-pad_width]

    image = util.montage(image_set,
                         grid_shape=grid_shape,
                         multichannel=multichannel)

    return image


def save_callbacks_csv(callbacks, filename_base='larson'):
    """Small utility function to save Keras's callbacks.
    """
    np.savetxt(f'{filename_base}-accuracy.csv',
               np.asarray(callbacks.history['accuracy']),
               delimiter=',')

    np.savetxt(f'{filename_base}-val_accuracy.csv',
               np.asarray(callbacks.history['val_accuracy']),
               delimiter=',')

    np.savetxt(f'{filename_base}-loss.csv',
               np.asarray(callbacks.history['loss']),
               delimiter=',')

    np.savetxt(f'{filename_base}-val_loss.csv',
               np.asarray(callbacks.history['val_loss']),
               delimiter=',')
    return None


def save_cropped_chunk(image, window_shape=(32, 32, 32), step=32,
                       folder='temp'):
    """Crops image and saves the cropped chunks in disk.

    Parameters
    ----------
    image : ndarray
        Input image.
    window_shape : integer or tuple of length image.ndim, optional
        (default : (32, 32, 32))
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length image.ndim, optional (default : 32)
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    folder : str, optional (default : 'temp')
        The folder to save the cropped files.

    Returns
    -------
        None
    """
    chunk_crop = np.vstack(np.hstack(
        util.view_as_windows(image,
                             window_shape=window_shape,
                             step=step)
    ))

    for idx, aux in enumerate(chunk_crop):
        fname = 'chunk_crop-%06d.tif' % (idx)
        io.imsave(os.path.join(folder, fname), aux)
    return None


def save_cropped_image(image, index, window_shape=(512, 512), step=512,
                       folder='temp'):
    """Crops image and saves the cropped chunks in disk.

    Parameters
    ----------
    image : ndarray
        Input image.
    index : int
        Reference number to saved files.
    window_shape : integer or tuple of length image.ndim, optional (default : (512, 512))
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length image.ndim, optional (default : 512)
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    folder : str, optional (default : 'temp')
        The folder to save the cropped files.

    Returns
    -------
        None
    """
    img_crop = np.vstack(util.view_as_windows(image,
                                              window_shape=window_shape,
                                              step=step))

    for idx, aux in enumerate(img_crop):
        fname = '%03d_img_crop-%03d.png' % (index, idx)
        io.imsave(os.path.join(folder, fname), aux)
    return None


def save_predictions(save_folder, predictions, multichannel=False,
                     num_class=2):
    for idx, pred in enumerate(predictions):
        if multichannel:
            output = _aux_label_visualize(image=pred,
                                          num_class=num_class,
                                          color_dict=const.COLOR_DICT)
        else:
            output = pred[:, :, 0]
        io.imsave(os.path.join(save_folder,
                               '%03d_predict.png' % (idx)),
                 util.img_as_ubyte(output))
    return None


def tensor_generator(images, multichannel=False):
    """
    """
    for image in images:
        image = image / 255
        if not multichannel:
            image = np.reshape(image, image.shape+(1,))
        image = np.reshape(image, (1,)+image.shape)

        yield image


def _assert_compatible(image_1, image_2):
    """Raise an error if the shape and dtype do not match."""
    if not image_1.shape == image_2.shape:
        raise ValueError('Input images do not have the same dimensions.')
    return None


def _assert_folder_exists(folder):
    """Raise an error if the folder does not exist."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"The folder '{folder}' does not exist.")
    return None


def _assert_same_length(data_1, data_2):
    """Raise an error if the data does not have the same length."""
    if not len(data_1) == len(data_2):
        raise ValueError('Input data do not have the same length.')
    return None


def _aux_label_visualize(image, color_dict, num_class=2):
    """
    """
    if np.ndim(image) == 3:
        image = image[..., 0]
    output = np.zeros(image.shape + (3,))
    for num in range(num_class):
        output[image == num, :] = color_dict[num]
    return output / 255


def _aux_predict(predictions, pad_width=16, grid_shape=(10, 10),
                 num_class=2, multichannel=False):
    """
    """
    aux_slice = len(grid_shape) * [slice(pad_width, -pad_width)]
    if multichannel:
        # multiply pad_width with everyone, except depth and colors
        output = np.zeros((predictions.shape[0],
                           *np.asarray(predictions.shape[1:-1])-2*pad_width,
                           predictions.shape[-1]))
        for idx, pred in enumerate(predictions):
            aux_pred = _aux_label_visualize(image=pred,
                                            num_class=num_class,
                                            color_dict=const.COLOR_DICT)
            output[idx] = aux_pred[(*aux_slice, slice(None))]
    else:
        output = predictions[(slice(None), *aux_slice, 0)]

    if output.ndim == 3:
        output = util.montage(output,
                              fill=0,
                              grid_shape=grid_shape,
                              multichannel=multichannel)
    elif output.ndim == 4:
        output = montage_3d(output,
                            grid_shape=grid_shape)
    return output


def _available_2d_nets(network, window_shape, n_class, preset_model):
    available_nets = {
        'tiramisu': tiramisu(input_size=(*window_shape, n_class),
                             preset_model=preset_model),
        'unet': unet(input_size=(*window_shape, n_class)),
    }
    return available_nets.get(network, None)


def _available_3d_nets(network, window_shape, n_class, preset_model):
    available_nets = {
        'tiramisu_3d': tiramisu_3d(input_size=(*window_shape, n_class),
                                   preset_model=preset_model),
        'unet_3d': unet_3d(input_size=(*window_shape, n_class)),
        'unet_3d_3': unet_3d_3(input_size=(*window_shape, n_class))
    }
    return available_nets.get(network, None)


def _folder_samples(base_ref, subfolder_ref):
    """Returns the folder structure for each sample.
    """
    folders = []
    for base_path, subfolders, _ in os.walk(base_ref):
        if subfolder_ref in subfolders:
            folders.append(os.path.join(base_path, subfolder_ref))
    return folders


def _suitable_samples(pred_folders, gold_folders):
    """Returns the path of each sample contained in both prediction and
    gold standard folders.
    """
    gold_samples = []

    pred_samples = [folder.split('/')[-2] for folder in pred_folders]

    # some folders in Larson's gold standard have a folder named 'Registered',
    # which contains more information that we need for comparison. We need a
    # special condition to be able to get the right path on these folders.
    for folder in gold_folders:
        aux = folder.split('/')
        if 'Registered' in aux:
            gold_samples.append(aux[-3])
        else:
            gold_samples.append(aux[-2])

    valid_samples = list(set(pred_samples) & set(gold_samples))

    pred_idx = [pred_samples.index(sample) for sample in valid_samples]
    gold_idx = [gold_samples.index(sample) for sample in valid_samples]

    pred_valid = [pred_folders[idx] for idx in pred_idx]
    gold_valid = [gold_folders[idx] for idx in gold_idx]

    return pred_valid, gold_valid
