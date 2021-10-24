import numpy as np
import os
import re
from natsort import natsorted
from glob import glob
import joblib
from scipy.fft import fft
from skimage import exposure

def fft_similarity(fft_1 : np.ndarray, fft_2 : np.ndarray):
    '''Computes the distance between 2 FFT matrices based on the L2 norm, focusing on extreme high and low frequencies'''

    score = np.linalg.norm(fft_1[15:83, 0:3] - fft_2[15:83, 0:3]) + np.linalg.norm(fft_1[15:83, -16:] - fft_2[15:83, -16:])
    return score


def load_dataset(folder_path : str = os.path.abspath(f'/Users/Corty/Downloads/fft_arrays/**/')):
    '''This function loads the FFT matrices based on the dataset. It takes a folder path as input,
    with 2 distinct subfolders for normal and anomalous samples:
    fft_normal_dice_arrays and fft_anomalous_dice_arrays'''

    data = {}
    anomalies = []
    for i in range(0,11):
        data[i] = []
        # natsorted preserves intuitive ordering for 1, 10, 100 etc.
        for path in natsorted(glob(folder_path, recursive=True)):
            pattern_1 = re.compile(f"arr_{i}/.+")
            if pattern_1.search(path):
                arr = np.load(path, allow_pickle=True)
                data[i].append(arr)

    for path in natsorted(glob(folder_path, recursive=True)):
        pattern_2 = re.compile(f"fft_anomalous_dice_arrays/.+")
        if pattern_2.search(path):
            arr = np.load(path, allow_pickle=True)
            anomalies.append(arr)

    return data, anomalies


def create_class_avg(data : dict):
    '''Creates a template for each class by averaging FFT matrices'''

    avg_dict = {}
    for _class, arr in data.items():
        avg_dict[_class] = sum(arr) / len(arr)
    return avg_dict


def predict_class(fft_test : np.ndarray, models : dict):
    '''Checks distances based on fft_similarity between a given FFT matrix and the templates.
    Returns a tuple (distance, class, all_distances)'''

    scores = []
    for _class, fft in models.items():
        scores.append((fft_similarity(fft_test, fft)))
    return np.min(scores), np.argmin(scores), scores


def create_metrics(data : dict, models : dict, precision_rate : float):
    '''Creates thresholds that will be used to determine when a FFT matrix contains an anomaly.
    The threshold is the highest distance found between clean samples of each class and the corresponding
    template, adjusted by a precision_rate coefficient that makes the prediction less precise but potentially
    gives a higher recall'''

    max_dict = {}
    for _class, arr in data.items():
        _max = 0
        score = 0
        for fft in arr:
            if predict_class(fft, models):
                score = predict_class(fft, models)[0]
            if score > _max:
                _max = score
        max_dict[_class] = _max * precision_rate
    return max_dict


def predict_anomaly(fft_test : np.ndarray, models : dict, threshold_dict : dict):
    '''This function tries to decide whether a given FFT matrix contains an anomaly
    based on the relevant templates and thresholds'''

    input_score, predicted_class, class_scores = predict_class(fft_test, models)
    if input_score > threshold_dict[predicted_class]:
        return True
    else:
        return False


def preprocess_input(image : np.ndarray) -> np.ndarray:
    '''Processes the image by cropping and slightly changing the exposure
    to make the detection easier. Returns the FFT of that image.'''

    data = np.copy(np.asarray(image))
    data = exposure.adjust_gamma(data, gamma=1.1, gain=1.001)
    data = exposure.adjust_log(data, gain=1.001)
    data = data[15:113, 15:113]
    return abs(fft(data))


def fft_detector(image : np.ndarray, predictive_strength=0.9):
    '''Main function that takes as input an image, as well as a predictive_strength parameter
    that will pass a value between 0 and 1 to the precision_rate parameter of create_metrics.
    A value close to 0 makes the detection highly sensitive and imprecised, while close to 1
    makes it less sensitive and more precise. Returns whether an anomaly was detected or not,
    the predicted class and the false positives associated with the predictive strength'''

    fft = np.copy(np.asarray(image))
    data, anomalies = joblib.load("utils/data.pkl"), joblib.load("utils/anomalies.pkl")
    models = joblib.load("utils/models.pkl")
    thresholds_dict = create_metrics(data, models, predictive_strength)

    detected_anomaly = predict_anomaly(fft, models, thresholds_dict)
    detected_class = predict_class(fft, models)[1]

    false_positives_on_training_set = {}

    for idx in range(0, 11):
        false_positives_count = 0
        for fft in data[idx]:
            false_positives_count += predict_anomaly(fft, models, thresholds_dict)
        false_positives_on_training_set[idx] = f"{idx} : {false_positives_count}/{len(data[idx])}"

    # Example: True, 3, {0: 2/67, 1: 3/102, ...}
    return detected_anomaly, detected_class, false_positives_on_training_set