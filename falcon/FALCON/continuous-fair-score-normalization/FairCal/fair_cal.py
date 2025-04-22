import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from .beta_cal import BetaCalibration
from helper_classes.fairness_approach import Fairness_Approach

from helper_classes.result import Result
from tools.enums import Dataset, FRSystem, Method
from helper_classes.dataset import FRDataset

class FairCal (Fairness_Approach):
    method : Method = Method.FAIRCAL
    def __init__(self, 
        train_dataset: FRDataset, 
        test_datasets: list[FRDataset],
        n_clusters: int = 100,
        test_fmrs = [0.001]
        ):
        super().__init__(train_dataset, test_datasets, test_fmrs = test_fmrs)
        self.n_clusters = n_clusters
        self.subgroups = train_dataset.subgroups
        self._kmeans = None   
        self._stats = None
        self._cluster_calibration_method = None
        
    def train(self):
        clusters, self._stats, self._kmeans = self.cluster_methods(self.n_clusters)

        # Calculate Calibration Map (Beta Calibration) for each Cluster
        self._cluster_calibration_method = {}
        for i_cluster in tqdm(range(self.n_clusters)):
            scores_cal = clusters[i_cluster]['scores']
            ground_truth_cal = clusters[i_cluster]['ground_truth']
            # Cluster Calibration Map is the Beta Calibration based on the scores and the labels in the cluster
            self._cluster_calibration_method[i_cluster] = BetaCalibration(scores_cal, ground_truth_cal)

    def test(self):
        metrics = {}
        for test_dataset in self._test_datasets:
            ground_truth, scores, indices = test_dataset.comparison_scores(return_indices=True)
            predicted_clusters = self._kmeans.predict(test_dataset.embeddings)
            cluster_assignments = predicted_clusters[indices]

            confidences = np.zeros(len(scores))
            p = np.zeros(len(scores))
            for i_cluster in range(self.n_clusters):
                for t in [0, 1]:
                    select = cluster_assignments[:, t] == i_cluster
                    aux = scores[select] # aux = scores of clusters
                    if len(aux) > 0:
                        aux = self._cluster_calibration_method[i_cluster].predict(aux) # aux = prediction of beta calibration for this cluster with the given values
                        confidences[select] += aux * self._stats[i_cluster] # stats = number of scores per cluster
                        p[select] += self._stats[i_cluster]
            confidences = confidences / p

            subgroup_mask = self.get_subgroup_masks(test_dataset,indices)
            metrics[test_dataset.dataset] =  self.evaluate(test_dataset, ground_truth, confidences, subgroup_mask)
        return metrics
    
    def save_results(self, metrics, test_dataset:FRDataset, fmr):
        CWD = os.path.abspath(os.getcwd())
        parent_folder = os.path.join(os.path.join(CWD, self.method.value),"results")
        folder = os.path.join(os.path.join(parent_folder,self._fr_system.value),self._train_dataset.dataset.value+"-"+test_dataset.dataset.value)
        if not os.path.exists(folder):
                os.makedirs(folder)
        metrics.save(os.path.join(folder, f"fmr={fmr}"))   

    @staticmethod
    def get_result_from_file(fr_system : FRSystem, train_database : Dataset, test_dataset: Dataset, test_fmr: float = 0.001):
        CWD = os.path.abspath(os.getcwd())  # Current working directory
        parent_folder = os.path.join(os.path.join(CWD, FairCal.method.value),"results")     
        fr_folder = os.path.join(parent_folder,fr_system.value)
        dataset_folder = os.path.join(fr_folder,train_database.value+"-"+test_dataset.value)
        file_name = os.path.join(dataset_folder,"fmr="+str(test_fmr))
        if os.path.exists(file_name): return Result.load(file_name)    
        print(file_name+" does not exist")
        return None

    def cluster_methods(self, n_clusters):
        # KMeans Fit
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(self._train_dataset.embeddings)
        # Find Cluster assignments
        ground_truth, scores, train_indices = self._train_dataset.comparison_scores(return_indices=True)
        predicted_clusters = kmeans.predict(self._train_dataset.embeddings)
        cluster_assignments = predicted_clusters[train_indices]     

        # Setup clusters
        clusters = {} 
        for i_cluster in range(n_clusters):
            clusters[i_cluster] = {'scores': [], 'ground_truth': []}

        # Save for each cluster the scores, the labels of the comparisons of pairs where at least one is in the cluster
        stats = np.zeros(n_clusters)
        for i_cluster in range(n_clusters):
            select = np.logical_or(cluster_assignments[:, 0] == i_cluster, cluster_assignments[:, 1] == i_cluster)
            clusters[i_cluster]['scores'] = scores[select]
            clusters[i_cluster]['ground_truth'] = ground_truth[select]
            stats[i_cluster] = len(clusters[i_cluster]['scores'])
    
        return clusters, stats, kmeans

