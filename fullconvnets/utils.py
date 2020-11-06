from models.unet import unet, unet_3d
from models.tiramisu import tiramisu, tiramisu_3d
from skimage import color, exposure, io, util
from sklearn.metrics import auc, roc_curve
from tensorflow.keras.backend import clear_session

import constants as const
import csv
import evaluation
import numpy as np
import os


def check_path(pathname):
    """Check if the input path exists. If not, create it.

    Parameters
    ----------
    pathname : str
        The input path.

    Returns
    -------
    None
    """
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    return None


def check_tiramisu_layers(tiramisu_model=None):
    """Checks and returns the tiramisu layers on the tiramisu model string.

    Parameters
    ----------
    tiramisu_model : str or None, optional (default : None)
        The tiramisu model to be used.

    Returns
    -------
    tiramisu_layers : str
        String with the amount of layers. Empty string if tiramisu_model is
        None.
    """
    if tiramisu_model is not None:
        tiramisu_layers = tiramisu_model[8:]
    else:
        tiramisu_layers = ''
    return tiramisu_layers


def imread_prediction(image, is_binary=True):
    """Auxiliary function intended to be used with skimage.io.ImageCollection.

    Parameters
    ----------
    image : (M, N) array
        The input image.
    is_binary : boolean, optional (default : True)
        If True, returns a binary prediction image: True when pixel in the image
        is higher than 0.5, False when lower than or equal to 0.5.

    Returns
    -------
    prediction : (M, N) array
        The prediction image.
    """
    if is_binary:
        return util.img_as_float(io.imread(image)) > 0.5
    else:
        return util.img_as_float(io.imread(image))


def imread_goldstd(image):
    """Auxiliary function intended to be used with skimage.io.ImageCollection.
    Helps to read and process Larson et al's gold standard images.

    Parameters
    ----------
    image : (M, N) array
        The input image.

    Returns
    -------
    gold_standard : (M, N) array
        The gold standard image.
    """
    return process_goldstd_images(io.imread(image))


def measure_all_coefficients(data_test, data_gt,
                             calc_coef={'matthews': True, 'dice': True},
                             save_coef=True,
                             filename='coefficients.csv'):
    """Measures all comparison coefficients between two input data.

    Parameters
    ----------
    data_test : (M, N, P) array
        The input test data.
    data_gt : (M, N, P) array
        The gold standard data.
    calc_coef : dict, optional (default : {'matthews': True, 'dice': True})
        Determines what coefficients to calculate.
    save_coef : boolean, optional (default : True)
        If True, saves the coefficients to a file in the disk.
    filename : str, optional (default : 'coefficients.csv')
        If save_coef is True, where to save the calculated coefficients.

    Returns
    -------
    all_matthews : array
        Array containing all Matthews coefficient values for the input data.
    all_dice : array
        Array containing all Dice coefficient values for the input data.

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


def measure_roc_and_auc(data_pred, data_gs,
                        save_coef=True,
                        filename='coef_roc_and_auc.csv'):
    """Measures the ROC curve and its area under curve for the input data and
    a comparison gold standard.

    Parameters
    ----------
    data_pred : (M, N, P) array
        The input prediction data.
    data_gs : (M, N, P) array
        The gold standard data.
    save_coef : boolean, optional (default : True)
        If True, saves the coefficients to a file in the disk.
    filename : str, optional (default : 'coef_roc_and_auc.csv')
        If save_coef is True, where to save the calculated coefficients.

    Returns
    -------
    fpr_mean : array
        Mean and standard deviation for the false positive rates.
    tpr_mean : array
        Mean and standard deviation for the true positive rates.
    auc(fpr_mean[0], tpr_mean[0]) : array
        Area under curve for the mean ROC (true positive rate vs. false positive
        rate) curve.

    Example
    -------
    >>> from skimage import io
    >>> data_bin = io.ImageCollection(load_pattern='res_figures/binary/Test_TIRR_0_1p5_B0p2_*_bin.png',
                                      plugin=None)[1000:2000]
    >>> data_gs = io.ImageCollection(load_pattern='gt_figures/19_Gray_*.tif',
                                     plugin=None)
    >>> tp_rate, fp_rate, area_curve = measure_roc_and_auc(data_bin,
                                                           data_gs,
                                                           save_coef=True)
    """
    roc_curves = []
    _assert_same_length(data_pred, data_gs)

    for idx, (img_pred, img_gs) in enumerate(zip(data_pred, data_gs)):
        img_gs = process_goldstd_images(img_gs)
        _assert_compatible(img_pred, img_gs)

        aux_fpr, aux_tpr, _ = roc_curve(img_gs.ravel(), img_pred.ravel(),
                                        pos_label=1, drop_intermediate=False)

        roc_curves.append([aux_fpr, aux_tpr])

    roc_curves = np.asarray(roc_curves)
    fpr_mean = (roc_curves[:, 0].mean(axis=0),
                roc_curves[:, 0].std(axis=0))
    tpr_mean = (roc_curves[:, 1].mean(axis=0),
                roc_curves[:, 1].std(axis=0))

    if save_coef:
        with open(filename, 'a+') as file_coef:
            coef_writer = csv.writer(file_coef, delimiter=',')
            coef_writer.writerow(['fpr', fpr_mean[0], fpr_mean[1]])
            coef_writer.writerow(['tpr', tpr_mean[0], tpr_mean[1]])
            coef_writer.writerow(['auc', auc(fpr_mean[0], tpr_mean[0])])

    return fpr_mean, tpr_mean, auc(fpr_mean[0], tpr_mean[0])


def montage_3d(array_input, fill='mean', rescale_intensity=False,
               grid_shape=None, padding_width=0, multichannel=False):
    """Create a montage of several single- or multichannel cubes.

    Parameters
    ----------
    array_input : ndarray
        An array representing an ensemble of K images of equal shape.
    fill : float or array-like of floats or ‘mean’, optional (default : 'mean')
        Value to fill the padding areas and/or the extra tiles in the output
        array. Has to be float for single channel collections. For multichannel
        collections has to be an array-like of shape of number of channels. If
        mean, uses the mean value over all images.
    rescale_intensity : boolean, optional (default : False)
        Whether to rescale the intensity of each image to [0, 1].
    grid_shape : tuple, optional (default : None)
        The desired grid shape for the montage (ntiles_row, ntiles_column,
        ntiles_plane). The default aspect ratio is cubic.
    padding_width=0 : int, optional (default : 0)
        The size of the spacing between the tiles and between the tiles and the
        borders. If non-zero, makes the boundaries of individual images easier
        to perceive.
    multichannel : boolean, optional (default : False)
        If True, the last dimension is threated as a color channel, otherwise as
        spatial.
    """
    if multichannel:
        array_input = np.asarray(array_input)
    else:
        array_input = np.asarray(array_input)[..., np.newaxis]

    if array_input.ndim != 5:
        raise ValueError('Input array has to be either 4- or 5-dimensional')

    n_images, n_planes, n_rows, n_cols, n_channels = array_input.shape
    if grid_shape:
        ntiles_plane, ntiles_row, ntiles_col = [int(s) for s in grid_shape]
    else:
        ntiles_plane = ntiles_row = ntiles_col = int(np.ceil(np.sqrt(n_images)))

    # Rescale intensity if necessary
    if rescale_intensity:
        for idx in range(n_images):
            array_input[idx] = exposure.rescale_intensity(array_input[idx])

    # Calculate the fill value
    if fill == 'mean':
        fill = array_input.mean(axis=(0, 1, 2, 3))
    fill = np.atleast_1d(fill).astype(array_input.dtype)

    array_out = np.empty((
        (n_planes + padding_width) * ntiles_plane + padding_width,
        (n_rows + padding_width) * ntiles_row + padding_width,
        (n_cols + padding_width) * ntiles_col + padding_width,
        n_channels),
                         dtype=array_input.dtype)

    for idx_chan in range(n_channels):
        array_out[..., idx_chan] = fill[idx_chan]

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
        array_out[slices_plane[idx_sp], slices_row[idx_sr], slices_col[idx_sc], :] = image

    if not multichannel:
        array_out = array_out[..., 0]

    return array_out


def network_models(network='unet', window_shape=(288, 288), n_class=1,
                   preset_model='tiramisu-67'):
    """Returns a network model with window shape and number of classes defined.

    Parameters
    ----------
    network : str, optional (default : 'unet')
        Name of the network to be used.
    window_shape : (M, N) or (M, N, P) list, optional (default : (288, 288))
        Size of the window used on the network.
    n_class : int, optional (default : 1)
        Number of classes.
    preset_model : str, optional (default : 'tiramisu-67')
        Tiramisu preset model.

    Returns
    -------
    model : TensorFlow model
        A TensorFlow model defined according to the parameters.

    Notes
    -----
    window_shape receives two-dimensional or three-dimensional inputs,
    chosen to match the network.

    preset_model can receive tiramisu-56 and tiramisu-67; not applied when
    using U-nets.
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
                     window_shape=(32, 32, 32), step=16):
    """Returns a network prediction on a data chunk — i.e., a 3D object.

    Parameters
    ----------
    data : ndarray
        The input data.
    weights : TensorFlow weights
        A file with TensorFlow weights for the desired network.
    network : str (default : 'unet_3d')
        The network to use in prediction, corresponding to the weights file.
    n_class : int (default : 1)
        Number of classes to predict.
    pad_width : int (default : 16)
        Width of the padding to add in the borders of data.
    window_shape : (M, N, P) list (default : (32, 32, 32))
        Size of the window to process.
    step : int (default : 16)
        Size of the step used to chunk the data. Used as an overlap feature.

    Returns
    -------
    chunk_predictions : ndarray
        Array containing predict windows on the input data.
    """
    results = []
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
    data_size = data_crop.shape[0]
    crop_steps = 640

    for idx in range(0, data_size, crop_steps):
        # passing less objects at a time; trying to save some memory
        slice = data_crop[idx:idx+crop_steps]
        steps = slice.shape[0]
        chunk_gen = tensor_generator(slice)
        results.append(model.predict(chunk_gen, steps=steps, verbose=1))
        clear_session()
    return np.vstack(np.asarray(results))


def predict_on_image(image, weights, network='unet', n_class=1, pad_width=16,
                     window_shape=(288, 288), step=256):
    """Returns a network prediction on an image — i.e., a 2D object.

    Parameters
    ----------
    image : (M, N) array
        The input data.
    weights : TensorFlow weights
        A file with TensorFlow weights for the desired network.
    network : str, optional (default : 'unet')
        The network to use in prediction, corresponding to the weights file.
    n_class : int, optional (default : 1)
        Number of classes to predict.
    pad_width : int, optional (default : 16)
        Width of the padding to add in the borders of data.
    window_shape : (M, N) list, optional (default : (288, 288))
        Size of the window to process.
    step : int, optional (default : 16)
        Size of the step used to chunk the data. Used as an overlap feature.

    Returns
    -------
    prediction : (M, N) array
        Array containing predictions on the input data.
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


def prediction_folder(network='unet', tiramisu_model=None):
    """Returns the saving folder according to the network.

    Parameters
    ----------
    network : str, optional (default : 'unet')
        The network to be used.
    tiramisu_model : str or None, optional (default : None)
        The tiramisu model to be used.

    Returns
    -------
    pred_folder : str
        The folder predictions will be stored, according to
        constants.FOLDER_PRED_<NETWORK>.
    """
    tiramisu_layers = check_tiramisu_layers(tiramisu_model=tiramisu_model)

    available_folders = {
        'tiramisu': f'{const.FOLDER_PRED_TIRAMISU}{tiramisu_layers}',
        'tiramisu_3d': f'{const.FOLDER_PRED_TIRAMISU_3D}{tiramisu_layers}',
        'unet': const.FOLDER_PRED_UNET,
        'unet_3d': const.FOLDER_PRED_UNET_3D,
    }
    return available_folders.get(network, None)


def process_goldstd_images(data_goldstd):
    """Reads a gold standard image from Larson et al., returning only the
    fibers within it.

    Parameters
    ----------
    data_goldstd : (M, N) array
        Input image from Larson et al.

    Returns
    -------
    bin_goldstd : (M, N) array
        Binary image containing fibers detected by Larson et al.'s algorithm.
    """
    if np.unique(data_goldstd).size > 2:
        data_goldstd = data_goldstd == 217

    return util.img_as_bool(data_goldstd)


def process_sample(folder, data, weights, network='unet'):
    """Process the sample, predicting with the chosen network, and overlaps the
    original image with the results.

    Parameters
    ----------
    folder : str
        Folder where the results of prediction and overlap will be saved.
    data : (M, N, P) array
        The input data to be processed.
    weights : TensorFlow weights
        A file with TensorFlow weights for the desired network.
    network : str, optional (default : 'unet')
        The network to be used.

    Returns
    -------
    None

    Notes
    -----
    All results are saved in disk; please check if you have sufficient space to
    store the processed data.
    """
    FOLDER_PRED = os.path.join(folder, const.SUBFOLDER_PRED)
    FOLDER_OVER = os.path.join(folder, const.SUBFOLDER_OVER)

    for aux in [FOLDER_PRED, FOLDER_OVER]:
        check_path(pathname=aux)

    if network in const.AVAILABLE_3D_NETS:
        data = data.concatenate()
        n_planes, n_rows, n_cols = data.shape

        last_original_plane = n_planes
        # data should be divisible by const.STEP_3D.
        if n_rows > last_original_plane:
            data_complete = np.zeros((n_rows - n_planes, n_rows, n_cols),
                                     dtype='bool')
            # filling data if necessary, to ensure np.split works.
            data = np.concatenate((data, data_complete))
            n_planes, n_rows, n_cols = data.shape

        sections = n_planes / const.STEP_3D
        data = np.split(data,
                        indices_or_sections=sections)
        for idx_chunk, chunk in enumerate(data):
            # generating a list of possible filenames...
            filenames = []
            for num in range((idx_chunk)*const.STEP_3D,
                             (idx_chunk+1)*const.STEP_3D):
                if num < last_original_plane:
                    filenames.append('%06d.png' % (num))

            # ... then checking if the files exist, before processing a chunk.
            all_files_exist = all([os.path.isfile(
                os.path.join(FOLDER_PRED, aux_file))
                                   for aux_file in filenames])
            if not all_files_exist:
                results = predict_on_chunk(chunk,
                                           weights=weights,
                                           network=network,
                                           pad_width=const.PAD_WIDTH_3D,
                                           window_shape=const.WINDOW_SHAPE_3D,
                                           step=const.STEP_3D)
                # determining grid_shape.
                grid_shape = (1,
                              n_rows * 2 / const.WINDOW_SHAPE_3D[1],
                              n_cols * 2 / const.WINDOW_SHAPE_3D[2])

                prediction = _aux_predict(results,
                                          grid_shape=grid_shape)

                for idx_plane, plane in enumerate(prediction):
                    current_plane = idx_plane + idx_chunk*const.STEP_3D
                    filename = '%06d.png' % (current_plane)
                    # avoiding to save auxiliary slices with no info.
                    if current_plane < last_original_plane:
                        _check_and_save_prediction(folder=FOLDER_PRED,
                                                   prediction=plane,
                                                   filename=filename)
                        _check_and_save_overlap(folder=FOLDER_OVER,
                                                data_original=chunk[idx_plane],
                                                prediction=plane,
                                                filename=filename)
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
                _check_and_save_prediction(folder=FOLDER_PRED,
                                           prediction=prediction,
                                           filename=filename)

                # if file doesn't exist, overlaps the results on the original.
                _check_and_save_overlap(folder=FOLDER_OVER,
                                        data_original=image,
                                        prediction=prediction,
                                        filename=filename)

                clear_session()  # resetting TensorFlow session state.
    return None


def read_data(sample, folder_prediction, is_registered=False, is_binary=True):
    """Reads prediction data and its respective gold standard.

    Parameters
    ----------
    sample : dict
        Sample information according to Larson et al.'s data.
    folder_prediction : str
        Where the prediction data is in the disk.
    is_registered : boolean, optional (default : False)
        True if the sample has registered data available.
    is_binary : boolean, optional (default : True)
        If True, converts the input data into binary when reading.

    Returns
    -------
    data_prediction : (M, N, P) array
        The prediction data corresponding to sample.
    data_goldstd : (M, N, P) array
        The gold standard data corresponding to sample.
    """
    aux_folder = sample['folder']
    if is_registered:
        aux_folder += '_REG'
        aux_subfolder = const.SUBFOLDER_GOLDSTD_REG
    else:
        aux_subfolder = const.SUBFOLDER_GOLDSTD

    folder_pred = os.path.join(folder_prediction,
                               aux_folder,
                               const.SUBFOLDER_PRED,
                               f'*{const.EXT_PRED}')
    folder_goldstd = os.path.join(const.FOLDER_GOLDSTD,
                                  sample['folder'],
                                  aux_subfolder,
                                  f'*{const.EXT_GOLDSTD}')

    data_prediction = io.ImageCollection(load_pattern=folder_pred,
                                         load_func=imread_prediction,
                                         is_binary=is_binary)
    data_goldstd = io.ImageCollection(load_pattern=folder_goldstd,
                                      load_func=imread_goldstd)

    return data_prediction, data_goldstd


def _check_and_save_prediction(folder, prediction, filename):
    """
    
    Returns
    -------
    None
    """
    if not os.path.isfile(os.path.join(folder, filename)):
        io.imsave(os.path.join(folder, filename),
                  util.img_as_ubyte(prediction))
    return None


def _check_and_save_overlap(folder, data_original, prediction, filename):
    """

    Returns
    -------
    None"""
    if not os.path.isfile(os.path.join(folder, filename)):
        overlap = overlap_predictions(data_original, prediction)
        io.imsave(os.path.join(folder, filename),
                  util.img_as_ubyte(overlap))
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


def read_csv_roc_auc(filename):
    """Reads csv ROC and AUC saved in a file.
    
    Parameters
    ----------
    filename : str
    
    Returns
    -------
    fp_rate : ndarray
        Arrays containing mean and standard deviation of false positive rate for
        the processed samples.
    tp_rate : array
        Arrays containing mean and standard deviation of true positive rate for
        the processed samples.
    area_under_curve : float
        Area under curve obtained from fp_rate and tp_rate means.
    """
    coefs, fp, tp = [[] for _ in range(3)]
    csv_file = csv.reader(open(filename, 'r'))
    for row in csv_file:
        coefs.append(row)

    fp_rate, tp_rate, area_under_curve = coefs
    for (aux_fp, aux_tp) in zip(fp_rate, tp_rate):
        fp.append(aux_fp.replace('[', ' ').replace(']', ' ').split())
        tp.append(aux_tp.replace('[', ' ').replace(']', ' ').split())

    fp_rate = np.asarray(fp[1:], dtype='float')
    tp_rate = np.asarray(tp[1:], dtype='float')
    area_under_curve = np.asarray(area_under_curve[1:], dtype='float')

    return fp_rate, tp_rate, area_under_curve[0]


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
    """Small utility function to save TensorFlow callbacks.

        
    Returns
    -------
    None
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
    """Crops image and saves the cropped chunks in the disk.

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

    Notes
    -----
    Folder is created if it does not exist.
    """
    check_path(folder)

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
    """Crops image and saves the cropped chunks in the disk.

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
        io.imsave(os.path.join(folder, fname), util.img_as_ubyte(aux))
    return None


def save_predictions(save_folder, predictions, multichannel=False,
                     num_class=2):
    """
        
    Returns
    -------
    None
    """
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

        yield np.asarray(image, dtype='float32')


def _assert_compatible(image_1, image_2):
    """Raise an error if the shape of image_1 and image_2 do not match."""
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
                            fill=0,
                            grid_shape=grid_shape,
                            multichannel=multichannel)
    return output


def _available_2d_nets(network, window_shape, n_class, preset_model):
    """"""
    available_nets = {
        'tiramisu': tiramisu(input_size=(*window_shape, n_class),
                             preset_model=preset_model),
        'unet': unet(input_size=(*window_shape, n_class)),
    }
    return available_nets.get(network, None)


def _available_3d_nets(network, window_shape, n_class, preset_model):
    """"""
    available_nets = {
        'tiramisu_3d': tiramisu_3d(input_size=(*window_shape, n_class),
                                   preset_model=preset_model),
        'unet_3d': unet_3d(input_size=(*window_shape, n_class)),
    }
    return available_nets.get(network, None)


def _folder_samples(base_ref, subfolder_ref):
    """Returns the folder structure for each sample."""
    folders = []
    for base_path, subfolders, _ in os.walk(base_ref):
        if subfolder_ref in subfolders:
            folders.append(os.path.join(base_path, subfolder_ref))
    return folders


def _suitable_samples(pred_folders, gold_folders):
    """Returns the path of each sample contained in both prediction and
    gold standard folders."""
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
