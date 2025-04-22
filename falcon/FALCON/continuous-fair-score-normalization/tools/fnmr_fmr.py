import numpy as np
import tools.group_scores as groups

from tools.enums import *

import sklearn.metrics as metrics


def fnmr_at_fmr(labels: np.array, scores: np.array, fmr: float= 0.001, return_fpr_thr = False) -> tuple[float, float, float]:
    """
    Calculates the False Non Match Rate (FNMR) at the given False Match Rate (FMR) from the given comparison scores

    Parameters
    ----------
    genuine_scores : np.array
        The genuine comparison scores of the embeddings
    imposter_scores : np.array
        The imposter comparison scores of the embeddings
    fmr : float
        The False Match Rate at which the False Non Match Rate should be calculated
    roc_file : str
        The path to the image in the given face recognition system and dataset folder        

    Returns
    -------
    fnmr : float
        The False Non Match Rate (FNMR) at the given False Match Rate (FMR)
    actual_fmr : float 
        The actual False Match Rate (FMR) that is the smallest possible FMR from the data above the given FMR
    threshold : float
        The threshold at which the FNMR and the FMR come to be

    """
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores,drop_intermediate=False)
    index = np.argmin(np.abs(fpr - fmr))
    if return_fpr_thr: return [1 - tpr[index], fpr[index], thresholds[index]]
    else: return 1 - tpr[index]
    


def fnmrs_from_attribute_at_fmr(dataset, attribute: Attribute, fmr: float = 0.001) -> np.array:
    """
    Calculates the False Non Match Rates (FNMRs) at the given False Match Rate (FMR) for all classes of the given attribute, dataset and face recognition system

    Parameters
    ----------
    fr_system : FRSystem
        The face recognition system to use the embeddings from
    dataset : Dataset
        The dataset to use the embeddings from
    attribute : Attribute
        The label attribute for which the FNMRs of each class are to be calculated
    fmr : float
        The False Match Rate at which the FNMRs are to be calculated

    Returns
    -------
    fnmr_infos : np.array
        An tuple for each class of the given attribute, dataset and face recognition system which holds the following information
        (classname, FNMR, actual FMR)
        The actual False Match Rate (FMR) might be a bit of the given FMR. It is the smallest possible FMR from the data above the given FMR
    """
    fnmr_infos = []
    for attribute in groups.get_classes_of_attribute(dataset, attribute):
        subgroup_datset = dataset.get_subset(dataset.get_subgroup_mask(attribute))
        labels, scores = subgroup_datset.comparison_scores()
        fnmr, actual_fmr, thresholds = fnmr_at_fmr(
            labels, scores, fmr, return_fpr_thr=True)
        fnmr_infos.append((attribute, fnmr, actual_fmr))
    return fnmr_infos

def false_rates_at_threshold(labels, scores, threshold: float) -> tuple[float, float]:
    """
    Calculates the False Non Match Rate (FNMR) and the False Match Rate (FMR) from the given comparison scores at the given threshold

    Parameters
    ----------
    genuine_scores : np.array
        The genuine comparison scores of the embeddings
    imposter_scores : np.array
        The imposter comparison scores of the embeddings
    threshold : float
        If a comparison score is above this value, it is considered genuine

    Returns
    -------
    fnmr : float
        The False Non Match Rates (FNMRs) at the given threshold
    fmr : float
        The False Match Rates (FMRs) at the given threshold
    """
    score_above_thr = np.array([score >= threshold for score in scores], dtype=bool)
    fmr = np.sum(np.logical_and(score_above_thr == True,
                 labels == False))/scores.size
    fnmr = np.sum(np.logical_and(score_above_thr == False,
                  labels == True))/scores.size
    return fnmr, fmr