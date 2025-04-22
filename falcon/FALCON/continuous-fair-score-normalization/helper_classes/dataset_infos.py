import numpy as np
from dataset import FRDataset
from tools.enums import Attribute, Dataset, FRSystem

'''
This file provides information of the label distribution of the datasets
'''

indices = {
    Attribute.AGE:1, 
    Attribute.ETHNICS:2, 
    Attribute.GENDER:3
}

table = []
for dataset in [Dataset.ADIENCE,Dataset.COLORFERET, Dataset.LFW][2:]:
    fr_dataset = FRDataset(FRSystem.ARCFACE, dataset)
    for attribute,subgroup_dict in fr_dataset.grouped_subgroups_with_category.items():
        print("\nAttribute: "+attribute.value)
        for subgroup_name, subsubgroups in subgroup_dict.items():
            subgroup_mask = fr_dataset.get_subgroup_mask(subsubgroups)
            n = np.sum(subgroup_mask)
            print(subgroup_name+" & "+str(n)+" & "+str(round(n/len(fr_dataset)*100,2))+"\%")
        n = np.sum(fr_dataset.features[attribute] == "None")
        print("None & "+str(n)+" & "+str(round(n/len(fr_dataset)*100,2))+"\%")