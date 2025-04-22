import os
import pickle
from helper_classes.dataset import FRDataset
from helper_classes.fairness_approach import Fairness_Approach
from tools.fnmr_fmr import fnmr_at_fmr
from helper_classes.result import Result
from tools.enums import Dataset, FRSystem, Method
from .methods import KMeansClustering
from tqdm import tqdm
import numpy as np


class FSN(Fairness_Approach):
    method : Method = Method.FSN
    
    def __init__(self,
                train_dataset: FRDataset, 
                test_datasets: list[FRDataset], 
                k:int = 100,
                train_fmr = 0.001,
                test_fmrs = [0.001]
                 ):
        super().__init__(train_dataset, test_datasets, test_fmrs = test_fmrs)
        self._train_fmr = train_fmr
        self._method = KMeansClustering(k)
        self._global_thr = None, 
        self._cluster_results = None
        
    @property
    def clustering_method(self):
        return self._method
    
    def train(self):        
        self.clustering_method.reset()
        cluster_assignment, existing_clusters = self.clustering_method.fit(self._train_dataset)
        
        #Calculate Threshold per Cluster
        cluster_results = np.empty((self.clustering_method.cluster_amount,))
        
        for k in tqdm(
                        range(self.clustering_method.cluster_amount),
                        desc="Calculate Cluster"):
            # Cluster was removed due to reclustering
            if not existing_clusters[k]:
                cluster_results[k] = np.nan
                continue
            
            cluster_mask = cluster_assignment == k

            gen_imp_labels, scores = self._train_dataset.comparison_scores(mask = cluster_mask, return_indices=False)

            # Skip the calculation if the cluster is empty or it does not contain any imposter/genuine pairs.
            if len(gen_imp_labels) == 0 or len(np.unique(gen_imp_labels)) != 2:
                print(f"Invalid scores generated. Dataset {self._train_dataset.dataset_name}, k={k}")
                continue

            cluster_results[k] = fnmr_at_fmr(gen_imp_labels,scores, return_fpr_thr=True,fmr=self._train_fmr)[2] 

        # gen_imp_labels, scores = generate_scores(self._train_dataset, return_indices = False)
        gen_imp_labels, scores = self._train_dataset.comparison_scores(return_indices=False)
        global_thr = fnmr_at_fmr(gen_imp_labels,scores, return_fpr_thr=True,fmr=self._train_fmr)[2]

        self._global_thr = global_thr, 
        self._cluster_results = cluster_results
            
    def test(self):
        metrics = {}
        for dataset in self._test_datasets:
            cluster_assignment = self.clustering_method\
                                .predict_cluster(dataset.embeddings)                             
                                
            # gen_imp_labels, scores, row_indices = generate_scores(dataset, return_indices=True)
            gen_imp_labels, scores, row_indices = dataset.comparison_scores(return_indices=True)
            
            # Scores are now the normalized scores
            clusters_left = cluster_assignment[row_indices[:,0]]
            scores_left =  self._cluster_results[clusters_left] - self._global_thr
            clusters_right = cluster_assignment[row_indices[:,1]]
            scores_right =  self._cluster_results[clusters_right] - self._global_thr            
            scores_normed = scores - np.multiply(0.5, scores_left + scores_right)

            subgroup_masks = self.get_subgroup_masks(dataset,row_indices)

            metrics[dataset.dataset] = self.evaluate(dataset, gen_imp_labels, scores_normed, subgroup_masks)
        return metrics
    

    def save_results(self, metrics, test_dataset:FRDataset, fmr:float):
        CWD = os.path.abspath(os.getcwd())
        parent_folder = os.path.join(os.path.join(CWD, self.method.value),"results")
        fmr_folder = os.path.join(parent_folder,f"train_fmr={self._train_fmr}")
        folder = os.path.join(os.path.join(fmr_folder,self._fr_system.value),self._train_dataset.dataset.value+"-"+test_dataset.dataset.value)
        if not os.path.exists(folder):
                os.makedirs(folder) 
        metrics.save(os.path.join(folder, f"fmr={fmr}"))
    
    @staticmethod
    def get_result_from_file(fr_system : FRSystem, train_database : Dataset, test_dataset: Dataset, train_fmr: float = 0.001, test_fmr: float = 0.001):
        CWD = os.path.abspath(os.getcwd())  # Current working directory
        parent_folder = os.path.join(os.path.join(CWD, FSN.method.value),"results")     
        fmr_folder = os.path.join(parent_folder,f"train_fmr={train_fmr}")
        fr_folder = os.path.join(fmr_folder,fr_system.value)
        dataset_folder = os.path.join(fr_folder,train_database.value+"-"+test_dataset.value)
        file_name = os.path.join(dataset_folder,"fmr="+str(test_fmr))
        if os.path.exists(file_name): return Result.load(file_name)    
        print(file_name+" does not exist")
        return None