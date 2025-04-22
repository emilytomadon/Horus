import os
from helper_classes.dataset import FRDataset
from helper_classes.fairness_approach import Fairness_Approach
from helper_classes.result import Result
from tools.enums import Dataset, FRSystem, Method


class Baseline(Fairness_Approach):
    method : Method = Method.BASELINE
    def __init__(self,
                test_datasets: list[FRDataset],
                test_fmrs = [0.001]
                 ):
        super().__init__(None, test_datasets, test_fmrs = test_fmrs)

    def train(self):
        pass
        
    def test(self):
        metrics = {}
        for test_dataset in self._test_datasets:
            labels, scores, indices = test_dataset.comparison_scores(return_indices=True)
            subgroup_masks = self.get_subgroup_masks(test_dataset,indices)
            metrics[test_dataset.dataset] = self.evaluate(test_dataset,labels,scores,subgroup_masks)
        return metrics
    
    def save_results(self, metrics: Result, test_dataset:FRDataset, fmr):
        CWD = os.path.abspath(os.getcwd())
        parent_folder = os.path.join(os.path.join(CWD, self.method.value),"results")
        folder = os.path.join(os.path.join(parent_folder,test_dataset.fr_system.value),test_dataset.dataset.value)
        if not os.path.exists(folder):
                os.makedirs(folder)
        metrics.save(os.path.join(folder, f"fmr={fmr}"))
    
    @staticmethod
    def get_result_from_file(fr_system : FRSystem, test_dataset: Dataset, test_fmr: float = 0.001):
        CWD = os.path.abspath(os.getcwd())  # Current working directory
        parent_folder = os.path.join(os.path.join(CWD, Baseline.method.value),"results")     
        fr_folder = os.path.join(parent_folder,fr_system.value)
        dataset_folder = os.path.join(fr_folder,test_dataset.value)
        file_name = os.path.join(dataset_folder,"fmr="+str(test_fmr))
        if os.path.exists(file_name): return Result.load(file_name)    
        print(file_name+" does not exist")
        return None