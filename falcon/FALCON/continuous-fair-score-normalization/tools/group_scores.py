import os
import numpy as np
from tools.enums import Attribute

def get_all_attributes(dataset, full_path: bool = False) -> list:
    """
    Collects all attributes of labels which exist in the given face recognition system and dataset

    Parameters
    ----------
    fr_system : FRSystem
        The face recognition system to search for attributes
    dataset : Dataset
        The dataset to search for attributes
    full_path : bool
        Whether the paths to the attribute folders should be returned as well

    Returns
    -------
    list
        Either the names of the attributes in a list
        Or if full_path is set to true, the names of the attributes with the paths to the attribute folders in a tuple
    """
    dataset_folder = dataset.get_folder()
    attributes = {}
    for label_file in os.listdir(dataset_folder):
        if not label_file.startswith("labels_"):
            continue
        attribute_name = label_file.split('_')[1].split('.')[0].upper()
        if attribute_name not in Attribute._member_names_:
            raise ValueError("Attribute "+attribute_name+" not in Attributes")
        attributes[Attribute[attribute_name]] = os.path.join(
            dataset_folder, label_file)
    if not full_path:
        return attributes.keys()
    else:
        return attributes.items()


def get_attribute_path(dataset, attribute: Attribute) -> str:
    """
    Returns the path to the attribute label file of the given attribute in the given face recognition system and dataset

    Parameters
    ----------
    fr_system : FRSystem
        The face recognition system to search for the attribute
    dataset : Dataset
        The dataset to search for the attribute
    attribute : Attribute
        The attribute to search for

    Returns
    -------
    str
        The path to the attribute folder of the given attribute in the given face recognition system and dataset
    """
    dataset_folder = dataset.get_folder()

    attribute_path = os.path.join(dataset_folder, "labels_"+attribute.name.lower()+".npy")
    if not os.path.exists(attribute_path):
        raise ValueError("No file "+attribute_path+" exists")
    return attribute_path

def get_classes_of_attribute(dataset, attribute: Attribute) -> list:#, full_path: bool = False) -> list:
    """
    Collects all classes of the given label attribute in the given face recognition system and dataset

    Parameters
    ----------
    fr_system : FRSystem
        The face recognition system to search for classes in the attribute
    dataset : Dataset
        The dataset to search for classes in the attribute
    attribute : Attribute
        The attribute of which the classes should be returned
    full_path : bool
        Whether the paths to the embedding files per class should be returned as well

    Returns
    -------
    list
        Either the names of the classes in a list
        Or if full_path is set to true, the names of the classes with the paths to the embedding files of the class in a tuple
    """
    attribute_path = get_attribute_path(dataset, attribute)
    # Get the labels
    labels = np.load(attribute_path, allow_pickle=True)
    # The classes of the labels
    classes = np.unique(labels).tolist()
    if 'None' in classes:
        classes.remove('None')
    return classes