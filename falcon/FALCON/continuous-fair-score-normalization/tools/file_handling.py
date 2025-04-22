import os
from tools.enums import *

CWD = os.path.abspath(os.getcwd())  # Current working directory
DATA = os.path.join(CWD, "Data")


def get_folder(fr_system: FRSystem, dataset: Dataset) -> str:
    """    
    Builds and returns the folder of the data of the given face recognition system and dataset
    The path is build regarding the system the user uses

    Parameters
    ----------
    fr_system : FRSystem
        The face recognition system whose folder should be used
    dataset : Dataset
        The dataset of which the file should be used

    Returns
    -------
    str
        the path to the folder with the given parameters
    """
    fr_system_folder = os.path.join(
        DATA, fr_system.value)  # choose face recognition system folder
    dataset_folder = os.path.join(
        fr_system_folder, dataset.value)  # choose dataset folder
    return dataset_folder


def get_file(fr_system: FRSystem, dataset: Dataset, type: Datatype) -> str:
    """    
    Builds and returns the file ds_<<datatset>>_<<type>>.npy from the folder with the given fr_system name
    The path is build regarding the system the user uses

    Parameters
    ----------
    fr_system : FRSystem
        The face recognition system whose folder should be used
    dataset : Dataset
        The dataset of which the file should be used
    type : Datatype
        The type of while which should be returned: either the embeddings, the filenames or the identities

    Returns
    -------
    str
        the path to the file with the given parameters
    """

    return os.path.join(get_folder(fr_system, dataset), type.value + ".npy")