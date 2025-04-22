import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__))))


from tqdm import tqdm
from helper_classes.dataset import FRDataset
from tools.fairness_evaluation import fairness_scores
from tools.fnmr_fmr import false_rates_at_threshold, fnmr_at_fmr
from helper_classes.result import AttributeResult, Result
from tools.enums import Dataset, FRSystem, Method, Metric, Attribute
import numpy as np

class Fairness_Approach:
    method: Method = None

    def __init__(self, 
        train_dataset: FRDataset, 
        test_datasets: list[FRDataset], 
        name:str = "",
        test_fmrs: list[float]  = [0.001]
        ):
        self._train_dataset = train_dataset
        self._test_datasets = test_datasets
        self._name = name
        self._test_fmrs = test_fmrs
        if self._train_dataset != None and self._test_datasets != None and self.method != Method.SLF:
            self._fr_system = self._train_dataset.fr_system
            for test_dataset in self._test_datasets:
                assert(test_dataset.fr_system == self._fr_system)

    def train(self):
        raise ValueError("Needs to be implemented")

    def test(self):
        raise ValueError("Needs to be implemented")

    def save_results(self, metrics, test_dataset : FRDataset, fmr):
        raise ValueError("Needs to be implemented")
    
    @staticmethod
    def get_result_from_file(fr_system : FRSystem, train_database : Dataset, test_dataset: Dataset, test_fmr: float = 0.001):
        pass

    def evaluate(self, test_dataset: FRDataset, labels, scores, subgroup_masks):
        results = {}
        for fmr in self._test_fmrs:
            # Calculate FNMRs and global thresholds   
            fnmr, _, global_threshold = fnmr_at_fmr(labels, scores, return_fpr_thr=True, fmr=fmr)
            fr_system = test_dataset.fr_system  
            result = Result(self.method,self._train_dataset.dataset if self.method != Method.BASELINE else None, test_dataset.dataset,fr_system if self.method != Method.SLF else self._fr_systems, fmr, fnmr)
            for category in tqdm(test_dataset.grouped_subgroups_with_category.keys()):
                attribute_result = AttributeResult(category,accuracies={})
                # # Accuracy per subgroup  
                for subgroup in tqdm(test_dataset.grouped_subgroups_with_category[category].keys()):
                    subgroup_mask = subgroup_masks[category][subgroup]
                    attribute_result.add_accuracy(subgroup, false_rates_at_threshold(labels[subgroup_mask], scores[subgroup_mask], global_threshold)[0])

                attribute_result.fdr, attribute_result.ir, attribute_result.garbe = self.test_fairness(test_dataset, labels, scores, category, subgroup_masks[category], global_threshold)
                result.add_attribute_result(attribute_result)
                del attribute_result

            results[fmr] = result
            print(result)
            self.save_results(result, test_dataset, fmr)
        return results
    
    def test_fairness(self, dataset:FRDataset, labels, scores, category, subgroup_masks, global_threshold):
        print("Test fairness")
        fmrs_and_fnmrs = {"FMR":[], "FNMR":[]}
        for c in dataset.grouped_subgroups_with_category[category].keys():
            subgroup_mask = subgroup_masks[c]
            fnmrs, fmrs = false_rates_at_threshold(labels[subgroup_mask], scores[subgroup_mask], global_threshold)
            fmrs_and_fnmrs["FMR"].append(fmrs)
            fmrs_and_fnmrs["FNMR"].append(fnmrs)

        return fairness_scores(fmrs_and_fnmrs["FMR"], fmrs_and_fnmrs["FNMR"]) # fdr, ir, garbe

    def _indices_subgroup_mask(self, dataset: FRDataset, category : Attribute, subgroup:str, indices):
        """
        Filters the given indices tuples (e.g. of comparison scores) of ids of embeddings, that both embeddings are part of the given subgroup

        Parameters
        ----------
        dataset : FRDataset
            The dataset to use the embeddings from
        category : Attribute
            The attribute, the subgroup is from
        subgroup : str
            The subgroup all embeddings that should be filtered should be part of
        indices : np.array
            The indices that should be filtered
            That might be the indices tuples that correspond to comparison scores between pairs of embeddings
        
        Returns
        -------
        np.array
            A boolean array of the same length like indices, where the bits are set if the corresponding tuple in indices refers to ids that are both in the given subgroup
        """
        is_in_subgroup = dataset.features[category] == dataset.grouped_subgroups[subgroup][0]
        for i in range(1, len(dataset.grouped_subgroups[subgroup])):
            is_in_subgroup = np.logical_or(is_in_subgroup, dataset.features[category] == dataset.grouped_subgroups[subgroup][i])        
        return is_in_subgroup[indices[:,0]] & is_in_subgroup[indices[:,1]]

    def get_subgroup_masks(self, test_dataset:FRDataset, indices):
       return {
            category: {
                subgroup: self._indices_subgroup_mask(test_dataset, category, subgroup, indices)
                for subgroup in subgroups.keys()
            }
            for category, subgroups in test_dataset.grouped_subgroups_with_category.items()
        }

"""
-> Files:
{
    __description__: "
    PERFORMANCE: (x.xx, x.xx),
    AGE: {GARBE: (x.xx, x.xx), FDR: (x.xx, x.xx), IR: (x.xx, x.xx), PERFORMANCE: {"0-2": (x.xx, x.xx), "3-6": (x.xx, x.xx), ...}},
    GENDER: {GARBE: (x.xx, x.xx), FDR: (x.xx, x.xx), IR: (x.xx, x.xx), PERFORMANCE: {"Male": (x.xx, x.xx), "Female": (x.xx, x.xx)}},
    ETHNICS: {GARBE: (x.xx, x.xx), FDR: (x.xx, x.xx), IR: (x.xx, x.xx), PERFORMANCE: {"White": (x.xx, x.xx), "Black": (x.xx, x.xx), ...}},
}

"""