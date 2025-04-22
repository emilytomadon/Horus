import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics

from helper_classes.dataset import FRDataset
from helper_classes.fairness_approach import Fairness_Approach
from helper_classes.result import Result
from tools.enums import *
from tools.fnmr_fmr import *
import os

from sklearn.kernel_ridge import KernelRidge

class FALCON(Fairness_Approach):
    method : Method = Method.FALCON
    def __init__(self, 
        train_dataset: FRDataset, 
        test_datasets: list[FRDataset], 
        kernel: str = None, # Train
        regularization_alpha = None, 
        nearest_neighbors: int = None,
        train_fmr = 0.001,
        old_normalization: bool = False, # Test 
        omega: float  = 0.75,
        test_fmrs: list[float]  = [0.001]
        ):
        super().__init__(train_dataset, test_datasets, test_fmrs = test_fmrs)
        self._regularization_alpha = regularization_alpha
        self._degree = None

        if kernel is None:
            if self._fr_system == FRSystem.ARCFACE: kernel = "polynomial3"
            elif self._fr_system == FRSystem.FACENET: kernel = "linear"
            elif self._fr_system == FRSystem.MAGFACE: kernel = "polynomial3"
            elif self._fr_system == FRSystem.QMAGFACE: kernel = "rbf"
        
        if regularization_alpha is None:
            if self._fr_system == FRSystem.ARCFACE: regularization_alpha = 0.000001
            elif self._fr_system == FRSystem.FACENET: regularization_alpha = 0.1
            elif self._fr_system == FRSystem.MAGFACE: regularization_alpha = 10
            elif self._fr_system == FRSystem.QMAGFACE: regularization_alpha = 0.000001

        self._kernel = kernel
        if kernel.startswith("polynomial"):
            self._kernel_name = "polynomial"
            self._degree = int(kernel[-1])
        else: self._kernel_name = kernel
        self._krr = KernelRidge(regularization_alpha, kernel = self._kernel_name, degree=self._degree)
        self._nearest_neighbors = nearest_neighbors if nearest_neighbors == None or nearest_neighbors < len(train_dataset) else None
        self._train_dataset = train_dataset
        
        # self._test_datasets = test_datasets
        self._omega = omega
        self._old_normalization = old_normalization
        self._train_fmr = train_fmr
        self._test_fmrs = test_fmrs

        print(train_dataset.dataset_name, test_datasets[0].dataset_name, train_fmr, test_fmrs, kernel, regularization_alpha, old_normalization, omega)
    
    def _local_thresholds(self,dataset: FRDataset):
        # Find optimal local threshold by determine the the threshold at the given fmr
        cosine_similarities = metrics.pairwise.cosine_similarity(dataset.embeddings)
        opt_threshold_per_embedding = np.array([], dtype=float)
        for index in tqdm(range(len(dataset)), desc="Local Thresholds"):
            if self._nearest_neighbors == None:
                labels = dataset.ids == dataset.ids[index]
                opt_threshold_per_embedding = np.append(opt_threshold_per_embedding,fnmr_at_fmr(labels,cosine_similarities[index],return_fpr_thr = True,fmr=self._train_fmr)[2])
            else:
                sorted_indices = np.flip(np.argsort(cosine_similarities[index]))[:self._nearest_neighbors]
                labels = dataset.ids[sorted_indices] == dataset.ids[index]
                opt_threshold_per_embedding = np.append(opt_threshold_per_embedding,fnmr_at_fmr(labels,cosine_similarities[index][sorted_indices],return_fpr_thr = True,fmr=self._train_fmr)[2])
        assert opt_threshold_per_embedding.size == len(dataset), "The thresholds do not math the embeddings array"
        return opt_threshold_per_embedding
    

    def normalized_scores(self, unnormed_scores, indices, dataset : FRDataset, old_normalization: bool, omega: float):
        thresholds = self._krr.predict(dataset.embeddings)
        # delta_thresholds = thresholds - global_threshold
        if old_normalization:
            return unnormed_scores - 1/2 * (thresholds[indices[:,0]]+thresholds[indices[:,1]])
        else:
            return omega*unnormed_scores + (1-omega)*(1-(thresholds[indices[:,0]]+thresholds[indices[:,1]]))

    def train(self):
        self._krr.fit(self._train_dataset.embeddings,self._local_thresholds(self._train_dataset))

    def test(self):
        metrics = {}
        for test_dataset in self._test_datasets:

            # Calculate fnmrs for unnormalized scores
            labels, unnormed_scores, indices = test_dataset.comparison_scores(return_indices=True)

            # Get the masks for all subgroups
            subgroup_masks = self.get_subgroup_masks(test_dataset,indices)
            
            normed_scores = self.normalized_scores(unnormed_scores, indices, test_dataset, old_normalization=self._old_normalization, 
            omega=self._omega)
            metrics[test_dataset.dataset] = self.evaluate(test_dataset,labels,normed_scores,subgroup_masks)
        return metrics

    def save_results(self, metrics, test_dataset:FRDataset, fmr):
        CWD = os.path.abspath(os.getcwd())
        parent_folder = os.path.join(CWD, self.method.value,"results")
        fmr_folder = os.path.join(parent_folder,f"train_fmr={self._train_fmr}")
        folder = os.path.join(os.path.join(fmr_folder,self._fr_system.value),self._train_dataset.dataset.value+"-"+test_dataset.dataset.value)
        if not os.path.exists(folder):
                os.makedirs(folder)
        metrics.save(os.path.join(folder, f"fmr={fmr}"))
       
    
    @staticmethod
    def get_result_from_file(fr_system : FRSystem, train_database : Dataset, test_dataset: Dataset, train_fmr: float = 0.001, test_fmr: float = 0.001, kernel:str = None, alpha: float = None, omega: float = None):
        CWD = os.path.abspath(os.getcwd())  # Current working directory
        if kernel is None and alpha is None and omega is None:
            parent_folder = os.path.join(os.path.join(CWD, FALCON.method.value),"results")  
            fmr_folder = os.path.join(parent_folder,f"train_fmr={train_fmr}")
            fr_folder = os.path.join(fmr_folder,fr_system.value)        
            dataset_folder = os.path.join(fr_folder,train_database.value+"-"+test_dataset.value)
            file_name = os.path.join(dataset_folder,"fmr="+str(test_fmr))
            if os.path.exists(file_name): 
                return Result.load(file_name)    
            print(file_name+" does not exist")
        elif train_fmr == 0.001 and test_fmr == 0.001:
            file_name = os.path.join(CWD, FALCON.method.value,"parameter",fr_system.value,train_database.value+"-"+test_dataset.value,kernel, "alpha="+str(alpha)+"-omega="+str(omega))
            if os.path.exists(file_name): 
                return Result.load(file_name)    
            print(file_name+" does not exist")
        elif train_fmr == 0.001 and test_fmr == 0.0001:
            file_name = os.path.join(CWD, FALCON.method.value,"parameter_0001",fr_system.value,train_database.value+"-"+test_dataset.value,kernel, "alpha="+str(alpha)+"-omega="+str(omega))
            if os.path.exists(file_name): 
                return Result.load(file_name)    
            print(file_name+" does not exist")
        return None
        