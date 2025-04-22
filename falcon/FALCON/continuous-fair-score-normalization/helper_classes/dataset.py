import copy
import os
import numpy as np
import sklearn
from tqdm import tqdm
from typing import Tuple
import sys
from sklearn.metrics import pairwise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.group_scores import get_all_attributes, get_classes_of_attribute
from tools.enums import *

from scipy.spatial import distance


CWD = os.path.abspath(os.getcwd())  # Current working directory
DATA = os.path.join(CWD, "Data")

class FRDataset():
    def __init__(self, fr_system : FRSystem, dataset : Dataset):        
        if not isinstance(dataset, Dataset):
            raise ValueError(str(dataset)+ " is not a valid dataset")
        self.dataset = dataset        
        if not isinstance(fr_system, FRSystem):
            raise ValueError(str(fr_system)+ " is not a valid face recognition system")
        self.fr_system = fr_system

        self._embeddings = None
        self._ids = None
        self._filenames = None
        self._age_labels = None
        self._gender_labels = None
        self._ethnicity_labels = None

        self._features = None
        self._subgroups = None
        self._or_groups = None
        self._grouped_subgroups = None

    def __len__(self):
        return self.ids.shape[0]

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = np.load(self._get_data_file(Datatype.EMB))
        return self._embeddings
    
    @property
    def ids(self):
        if self._ids is None:
            self._ids = np.load(self._get_data_file(Datatype.IDENTITIES)).astype(int)
        return self._ids
    
    @property
    def filenames(self):
        if self._filenames is None:
            try:
                self._filenames = np.load(self._get_data_file(Datatype.FILENAMES),allow_pickle=True)                
            except:
                print("Warning: No filenames available of this Dataset")
                return None
        return self._filenames    

    @property
    def age_labels(self):
        if self._age_labels is None:
            self._age_labels = self._get_labels(Attribute.AGE)
        return self._age_labels

    @property
    def ethnicity_labels(self):
        if self._ethnicity_labels is None:
            self._ethnicity_labels = self._get_labels(Attribute.ETHNICS)
        return self._ethnicity_labels

    @property
    def gender_labels(self):
        if self._gender_labels is None:
            self._gender_labels = self._get_labels(Attribute.GENDER)
        return self._gender_labels
    
    @property
    def features(self):
        if self._features is None:
            feature_list = get_all_attributes(self)
            all_features = {Attribute.AGE : self.age_labels, Attribute.GENDER: self.gender_labels, Attribute.ETHNICS: self.ethnicity_labels}
            self._features = {feature_name : all_features[feature_name] for feature_name in feature_list}
        return self._features

    @property
    def subgroups(self):
        # {"Attribute.AGE": ["(00, 02)", "(04, 06),..."], " Attribute.GENDER": ["Male", "Female"], "Attribute.ETHNICS":...}
        if self._subgroups is None:    
            self._subgroups = {}
            for attribute in get_all_attributes(self):
                self._subgroups[attribute] =  get_classes_of_attribute(self, attribute)
        return self._subgroups

    @property
    def grouped_subgroups_with_category(self):
        if self._grouped_subgroups is None:
            if self.dataset == Dataset.COLORFERET:
                self._grouped_subgroups = {
                    Attribute.AGE: {'(0, 20)':['(0, 20)'], '(21, 30)':['(21, 30)'], '(31, 40)':['(31, 40)'], '(41, inf)':['(41, inf)']},
                    Attribute.ETHNICS:{'Asian':['Asian', 'Asian-Middle-Eastern', 'Asian-Southern'], 'Black':['Black-or-African-American'], 'Other':['Hispanic', 'Native-American', 'Other', 'Pacific-Islander'], 'White':['White']}, 
                    Attribute.GENDER:{'Female':['Female'], 'Male':['Male']}}
            elif self.dataset == Dataset.MORPH:
                self._grouped_subgroups = {
                     Attribute.AGE: {'(0, 20)':['(0, 20)'], '(21, 30)':['(21, 30)'], '(31, 40)':['(31, 40)'], '(41, inf)':['(41, inf)']},
                     Attribute.ETHNICS:{'Asian':['Asian'], 'African':['Black'], 'Hispanic':['Hispanic'], 'European':['White']},
                     Attribute.GENDER:{'Female':['Female'], 'Male':['Male']}}
            else:
                self._grouped_subgroups = {}
                for attribute, class_list in self.subgroups.items():
                    self._grouped_subgroups[attribute] = {}
                    for c in class_list:
                        self._grouped_subgroups[attribute][c] = [c]
            self._or_groups = np.ones(len(self._grouped_subgroups), dtype=bool)
        return self._grouped_subgroups
    
    @property
    def grouped_subgroups(self):
        # {'(0, 20)':['(0, 20)'], '(21, 30)':['(21, 30)'], '(31, 40)':['(31, 40)'], '(41, inf)':['(41, inf)'], 'Asian':['Asian', 'Asian-Middle-Eastern', 'Asian-Southern'], 'Black':['Black-or-African-American'], ...}
        return {subgroup: associated_labels for d in self.grouped_subgroups_with_category.values() for subgroup, associated_labels in d.items()}


    @property
    def or_groups(self):
        if self._or_groups is None:
            self.grouped_subgroups
        return self._or_groups
    
    @grouped_subgroups.setter
    def grouped_subgroups(self,grouped_subgroups, or_groups = True):
        """
        or_groups: bool|np.array(bool)
        """
        self._grouped_subgroups = grouped_subgroups
        if type(or_groups) is np.array:
            if or_groups.size != len(grouped_subgroups):
                raise ValueError(f"Operator list length {or_groups.size} does not match subgroup dictionary length {len(grouped_subgroups)}")
            self._or_groups = or_groups
        elif type(or_groups) is bool:
            self._or_groups = np.full(len(grouped_subgroups), or_groups)
        else:
            raise ValueError("Parameter or_groups is neither a boolean or a np.array of booleans")

    
    @property
    def dataset_name(self):
        return self.fr_system.value+"-"+ self.dataset.value

    def qMagFace_comparison_scores(self, embeddings = None) -> np.ndarray:
        if embeddings is None: embeddings = self.embeddings
        arr, quality = sklearn.preprocessing.normalize(embeddings, return_norm=True)
        # quality = np.minimum(quality,1)
        pairwise_qualities = np.minimum.outer(quality, quality)
        s = pairwise.cosine_similarity(embeddings)# s = pairwise.cosine_similarity(arr)
        ALPHA = 0.077428
        BETA = 0.125926
        omega = np.minimum(0, BETA * s - ALPHA)
        return omega * pairwise_qualities + s

    def comparison_scores(self, mask : np.array = None, return_indices = False):
        if len(self) <= 40000: return self.generate_scores(mask = mask, return_indices=return_indices)
        if return_indices: raise ValueError("Not implemented yet")
        genuine_scores, imposter_scores = self.generate_scores_large_data()
        return np.concatenate(
            [np.ones(len(genuine_scores), dtype=bool), np.zeros(len(imposter_scores), dtype=bool)]), np.concatenate(
            [np.array(genuine_scores), np.array(imposter_scores)])        

    def generate_scores(self, mask : np.array = None, return_indices : bool = False):
        if mask is None:
            mask = np.ones((len(self)), dtype=bool)
        elif np.sum(mask) == 0:
            if return_indices:
                return np.array([]),np.array([]),np.array([])
            return np.array([]),np.array([])
        
        embeddings = self.embeddings[mask]
        ids = self.ids[mask]
        cosine_similarities = pairwise.cosine_similarity(embeddings) if self.fr_system != FRSystem.QMAGFACE else self.qMagFace_comparison_scores(embeddings)

        ids_matrix = np.array([ids])
        genuine_indexes_matrix = np.equal(ids_matrix,np.transpose(ids_matrix))

        relevant_comparison_indexes = np.triu(np.ones((len(embeddings), len(embeddings)), dtype=bool), 1)
        if return_indices:
            return genuine_indexes_matrix[relevant_comparison_indexes], cosine_similarities[relevant_comparison_indexes], np.argwhere(relevant_comparison_indexes)
        return genuine_indexes_matrix[relevant_comparison_indexes], cosine_similarities[relevant_comparison_indexes]

    def generate_scores_large_data(self, imposter_count: int = 5) -> Tuple[np.array, np.array]:
        unique_ids = np.unique(self.ids)

        genuine_scores = np.array([])
        imposter_scores = np.array([])

        for id in tqdm(unique_ids):
            gen_ids = self.ids == id
            gen_ids_indices = np.arange(self.ids.size)[gen_ids]
            count = gen_ids_indices.size
            if count == 1: continue
            cos_sims = pairwise.cosine_similarity(self.embeddings[gen_ids_indices])        
            genuine_scores = np.append(genuine_scores,cos_sims[np.triu(np.ones((gen_ids_indices.size, gen_ids_indices.size), dtype=bool), 1)])

            number_imp_comparisons = count * (count-1) // 2 * imposter_count
            imp_ids_indices = np.arange(self.ids.size)[np.invert(gen_ids)]
            imp_count = imp_ids_indices.size
            if imp_count == 0:
                print("Warning: No imposter comparisons possible!")
                return [], []
            if number_imp_comparisons > count*imp_count:
                imposter_scores = np.append(imposter_scores, pairwise.cosine_similarity(self.embeddings[gen_ids_indices],self.embeddings[imp_ids_indices]))
                continue
            
            random_elements = np.random.choice(np.arange(count*imp_count),number_imp_comparisons,replace=False)

            random_id_indexes = random_elements % count
            random_not_id_indexes = (random_elements // count) % imp_count
            imposter_scores = np.append(imposter_scores, np.array([1-distance.cosine(self.embeddings[a],self.embeddings[b]) for a,b in list(zip(random_id_indexes,random_not_id_indexes))]))      

        return genuine_scores, imposter_scores

    def _get_labels(self, attribute : Attribute):
        dataset_folder = self.get_folder()
        attribute_path = os.path.join(dataset_folder, "labels_"+attribute.name.lower()+".npy")
        if os.path.exists(attribute_path):
            return np.load(attribute_path, allow_pickle=True)
        else:
            return None

    def _get_data_file(self, datatype : Datatype):
        if not isinstance(datatype, Datatype):
            raise ValueError(str(datatype)+ " is not a valid datatype")
        return os.path.join(self.get_folder(),datatype.value+".npy")

    def get_folder(self) -> str:
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
            DATA, self.fr_system.value)  # choose face recognition system folder
        dataset_folder = os.path.join(
            fr_system_folder, self.dataset.value)  # choose dataset folder
        return dataset_folder
    
    def apply_mask(self, mask):
        self._embeddings = self.embeddings[mask]
        self._ids = self.ids[mask]
        self._filenames = self.filenames[mask]
        if not self.age_labels is None:
            self._age_labels = self.age_labels[mask]
        if not self.gender_labels is None:
            self._gender_labels = self.gender_labels[mask]
        if not self.ethnicity_labels is None:
            self._ethnicity_labels = self.ethnicity_labels[mask]
    
    def get_subset(self, mask : np.array):
        subset_dataset = self.clone()
        subset_dataset.apply_mask(mask)
        return subset_dataset

    def appended_dataset(self, other):
        self._embeddings = np.append(self.embeddings, other.embeddings, axis=0)
        self._ids = np.append(self.ids, other.ids)
        if not self.age_labels is None:
            self._age_labels = np.append(self.age_labels, other.age_labels)
        if not self.gender_labels is None:
            self._gender_labels = np.append(self.gender_labels, other.gender_labels)
        if not self.ethnicity_labels is None:
            self._ethnicity_labels = np.append(self.ethnicity_labels, other.ethnicity_labels)
        return self

    def append_dataset(self, other):
        merged_dataset = self.clone()
        return merged_dataset.appended_dataset(other)
    
    def clone(self):
        return copy.deepcopy(self)

    def get_subgroup_mask(self, subgroup, or_labels : bool = True):
        """
            subgroup: str | list[str]
        """
        mask = np.zeros(len(self))
        if type(subgroup) is not list:
            subgroup = [subgroup]
        for s in subgroup:
            try:
                feature = next(key for key, value_list in self.subgroups.items() if s in value_list)
                if or_labels:
                    mask = np.logical_or(mask, self.features[feature] == s)
                else:
                    mask = np.logical_and(mask, self.features[feature] == s)
            except:
                raise ValueError("Subgroup "+s+" is not a subgroup of this dataset")
        
        return mask    

    def not_singleton_mask(self):
        uni_IDs, counts = np.unique(self.ids, return_counts=True)
        
        mask = np.zeros(self.__len__(), dtype=bool)
        for ID in uni_IDs[counts < 2]:
           mask = np.logical_or(mask, self.ids == ID)
        
        return np.logical_not(mask)
    
    def print_dataset_information(self):
        print("---"+self.dataset_name+"--- of length "+str(len(self)))
        uni_IDs, counts = np.unique(self.ids, return_counts=True)
        print("IDS:\t# Ids: "+str(uni_IDs.size)+", Average Quantity: "+str(np.round(np.average(counts),2))+", Max Quantity: "+str(np.max(counts)))
        print("EMBEDDINGS:\tShape: "+str(self.embeddings.shape))
        print("SUBGROUPS:\tGroups: "+str(self.subgroups)+", Grouped Subgroups: "+str(list(self.grouped_subgroups.keys())))
    
    def print_full_information(self):
        print("---"+self.dataset_name+"--- of length "+str(len(self)))
        uni_IDs, counts = np.unique(self.ids, return_counts=True)
        print("IDS:\t# Ids: "+str(uni_IDs.size)+", Average Quantity: "+str(np.round(np.average(counts),2))+", Max Quantity: "+str(np.max(counts)))
        print(self.ids)
        print("EMBEDDINGS:\tShape: "+str(self.embeddings.shape))
        print(self.embeddings)
        print("SUBGROUPS:\tGroups: "+str(self.subgroups)+", Grouped Subgroups: "+str(list(self.grouped_subgroups.keys())))