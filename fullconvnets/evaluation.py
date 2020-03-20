from fullconvnets import utils
from skimage import util

import numpy as np


def confusion_matrix(data_true, data_test):
    """Compares reference and test data to generate a confusion matrix.

    Parameters
    ----------
    data_true : ndarray
        Reference binary data (ground truth).
    data_test : ndarray
        Test binary data.

    Returns
    -------
    conf_matrix : array
        Matrix containing the number of true positives, false positives,
    false negatives, and true negatives.

    Notes
    -----
    The values true positive, false positive, false negative, and false
    positive are events obtained in the comparison between data_true and
    data_test:

                   data_true:             True                False
    data_test:
                            True       True positive   |   False positive
                                       ----------------------------------
                            False      False negative  |    True negative

    References
    ----------
    .. [1] Fawcett T. (2006) "An Introduction to ROC Analysis." Pattern
           Recognition Letters, 27 (8): 861-874, :DOI:`10.1016/j.patrec.2005.10.010`
    .. [2] Google Developers. "Machine Learning Crash Course with TensorFlow
           APIs: Classification: True vs. False and Positive vs. Negative."
           Available at:
           https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative
    .. [3] Wikipedia. "Confusion matrix." Available at:
           https://en.wikipedia.org/wiki/Confusion_matrix
    """
    _assert_compatible(data_true, data_test)

    true_pos = (data_true & data_test).sum() / data_true.size
    false_pos = (~data_true & data_test).sum() / data_true.size
    false_neg = (data_true & ~data_test).sum() / data_true.size
    true_neg = (~data_true & ~data_test).sum() / data_true.size

    return np.array([[true_pos, false_pos], [false_neg, true_neg]])


def difference_bin_gt(data_test, data_ref):
    """Returns where the resulting image and its ground truth differ.

    Parameters
    ----------
    data_test : ndarray
        Test binary data.
    data_ref : ndarray
        Reference binary data.

    Returns
    -------
    data_diff : ndarray
        Image showing where the resulting image and its ground truth
    differ.

    Notes
    -----
    The reference binary data is also known as ground truth or gold standard.
    """
    data_test = util.img_as_bool(data_test)
    data_ref = utils.process_goldstd_images(data_ref)

    return data_test ^ data_ref


def generate_heatmap(data_test, data_gt):
    """Generates a heatmap that summarizes the differences between all test images
    and their ground truths.

    Parameters
    ----------
    data_test : list, ImageCollection

    data_gt : list, ImageCollection

    Returns
    -------
    heatmap : ndarray
    """
    data_shape = data_test[0].shape
    heatmap = np.zeros(data_shape)

    for idx, img_test in enumerate(data_test):
        aux_diff = difference_bin_gt(img_test, data_gt[idx])
        heatmap += util.img_as_float(aux_diff / 255)

    return heatmap


def measure_dice(conf_matrix):
    """Calculate the Dice correlation coefficient for a confusion matrix.

    Parameters
    ----------
    conf_matrix : array
        Matrix containing the number of true positives, false positives,
    false_negatives, and true negatives.

    Returns
    -------
    coef_dice : float
        Dice correlation index for the input confusion matrix.

    Notes
    -----
    The Dice correlation coefficient is also know as F1-score.

    Dice and Jaccard correlation coefficients are equivalent:

        dice = 2 * jaccard / (1 + jaccard)

    The Tversky index [2]_ can be seen as a generalization of them both.

    References
    ----------
    .. [1] L. R. Dice, “Measures of the Amount of Ecologic Association Between
           Species,” Ecology, vol. 26, no. 3, pp. 297–302, Jul. 1945, doi:
           10.2307/1932409.
    .. [2] A. Tversky, “Features of similarity,” Psychological Review, vol. 84,
           no. 4, pp. 327–352, 1977, doi: 10.1037/0033-295X.84.4.327.
    """
    tr_pos, fl_pos, fl_neg, _ = conf_matrix.ravel()
    coef_dice = (2 * tr_pos) / (2 * tr_pos + fl_pos + fl_neg)
    return coef_dice


def measure_matthews(conf_matrix):
    """Calculate the Matthews correlation coefficient for a confusion matrix.

    Parameters
    ----------
    conf_matrix : array
        Matrix containing the number of true positives, false positives,
    false_negatives, and true negatives.

    Returns
    -------
    coef_matthews : float
        Matthews correlation index for the input confusion matrix.

    Notes
    -----
    Matthews correlation coefficient tends to be more informative than other
    confusion matrix measures, such as Dice coefficient and accuracy, when
    evaluating binary classification problems: it considers the balance ratios
    of true positives, true negatives, false positives, and false negatives.

    References
    ----------
    .. [1] Matthews B. W. (1975) "Comparison of the predicted and observed
           secondary structure of T4 phage lysozyme." Biochimica et Biophysica
           Acta (BBA) - Protein Structure, 405 (2): 442-451,
           :DOI:`10.1016/0005-2795(75)90109-9`
    .. [2] Chicco D. (2017) "Ten quick tips for machine learning in computational
           biology." BioData Mining, 10 (35): 35, :DOI:`10.1186/s13040-017-0155-3`
    .. [3] Wikipedia. "Matthews correlation coefficient." Available at:
           https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

    Examples
    --------
    >>> from skimage.measures import confusion_matrix
    >>> conf_matrix = confusion_matrix(data_true, data_test)
    >>> coef_matthews = measure_matthews(conf_matrix)
    """
    tr_pos, fl_pos, fl_neg, tr_neg = conf_matrix.ravel()
    coef_matthews = (tr_pos * tr_neg - fl_pos * fl_neg) / \
        np.sqrt((tr_pos + fl_pos) * (tr_pos + fl_neg) *
                (tr_neg + fl_pos) * (tr_neg + fl_neg))
    return coef_matthews


def _assert_compatible(image_1, image_2):
    """Raise an error if the shape and dtype do not match."""
    if not image_1.shape == image_2.shape:
        raise ValueError('Input images do not have the same dimensions.')
    return None
