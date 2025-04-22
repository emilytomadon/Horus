import os
from helper_classes.dataset import FRDataset
import numpy as np
from helper_classes.fairness_approach import Fairness_Approach
from helper_classes.result import Result

from tools.enums import Dataset, FRSystem, Method
from sklearn.preprocessing import MinMaxScaler

class SLF(Fairness_Approach):
    method : Method = Method.SLF
    def __init__(self,
        train_database: Dataset,
        test_databases: list[Dataset],
        fr_systems: list[FRSystem],
        align_train_datsets = False,
        test_fmrs = [0.001]
    ):
        assert len(fr_systems) >= 2
        fr_systems = sorted(fr_systems, key=lambda x: x.value)
        self._fr_systems = fr_systems
        self._train_database = train_database
        self._train_datasets = [FRDataset(fr_system, train_database) for fr_system in fr_systems]
        self._test_databases = test_databases
        super().__init__(FRDataset(fr_systems[0], train_database), [FRDataset(fr_systems[0], test_database) for test_database in test_databases], test_fmrs = test_fmrs)
        self._scalers = [MinMaxScaler()] * len(fr_systems)
        self._align_train_datsets = align_train_datsets

    @staticmethod
    def align_datasets(datasets: list[FRDataset]):
        filenames_list = [d.filenames for d in datasets]

        # Ensure that there is no duplicate in the filenames
        for filenames in filenames_list:
            assert len(np.unique(filenames)) == len(filenames)

        masks = []
        for i in range(len(filenames_list)):
            masks.append(np.ones(len(filenames_list[i])))
            for j in range(len(filenames_list)):
                if i == j: continue
                masks[i] = np.logical_and(masks[i], np.isin(filenames_list[i],filenames_list[j]))

        new_datasets = [datasets[i].get_subset(masks[i]) for i in range(len(filenames_list))]

        # Test that the resulting datasets have the same length
        lengths = [len(d) for d in new_datasets]
        assert np.max(lengths) == np.min(lengths)
        # Test that the resulting datasets have the data in the same order
        filenames = new_datasets[0].filenames
        for i in range(1,len(filenames_list)):
            assert np.array_equal(new_datasets[i].filenames, filenames)
        
        print(str([d.fr_system.value for d in datasets])+", "+str(datasets[0].dataset.value)+" : "+str([len(d) for d in datasets])+" -> "+str(len(new_datasets[0])))

        return new_datasets

    def train(self):
        for i in range(len(self._train_datasets)):
            self._scalers[i].fit(self._train_datasets[i].comparison_scores()[1].reshape(-1, 1))
    
    def test(self):
        metrics = {}
        for database in self._test_databases:
            test_datasets = SLF.align_datasets([FRDataset(fr_system, database) for fr_system in self._fr_systems])
            scores_list = []
            labels_list = []
            indices_list = []
            scaled_scores = []
            
            for i in range(len(test_datasets)):
                labels, scores, indices = test_datasets[i].comparison_scores(return_indices = True)
                scores_list.append(scores)
                labels_list.append(labels)
                indices_list.append(indices)
                scaled_scores.append(self._scalers[i].transform(scores.reshape(-1, 1)).reshape(1,-1)[0])

            number_scores = np.array([len(s) for s in scores_list])
            assert np.min(number_scores) == np.max(number_scores)
            
            labels = labels_list[0]
            for i in range(1,len(labels_list)):
                assert np.array_equal(labels_list[i], labels)
            
            indices = indices_list[0]
            for i in range(1,len(indices_list)):
                assert np.array_equal(indices_list[i], indices)
                
            fusioned_scores = np.sum(scaled_scores, axis = 0)

            subgroup_masks = self.get_subgroup_masks(test_datasets[0],indices_list[0])
            metrics[database] =  self.evaluate(test_datasets[0],labels_list[0],fusioned_scores,subgroup_masks)
        return metrics

    def save_results(self, metrics, test_dataset: FRDataset, fmr):
        CWD = os.path.abspath(os.getcwd())
        parent_folder = os.path.join(os.path.join(CWD, self.method.value),"results")
        fr_systems_names = np.sort(np.array([fr.value for fr in self._fr_systems]))
        folder_name = '-'.join(fr_systems_names)
        folder = os.path.join(os.path.join(parent_folder,folder_name),self._train_database.value+"-"+test_dataset.dataset.value)
        if not os.path.exists(folder):
                os.makedirs(folder)
        metrics.save(os.path.join(folder, f"fmr={fmr}"))

    @staticmethod
    def get_result_from_file(fr_systems : list[FRSystem], train_database : Dataset, test_dataset: Dataset, test_fmr: float = 0.001):
        fr_systems = sorted(fr_systems, key=lambda x: x.value)
        fr_systems_string = '-'.join([fr_system.value for fr_system in fr_systems])
            
        CWD = os.path.abspath(os.getcwd())  # Current working directory
        parent_folder = os.path.join(os.path.join(CWD, SLF.method.value),"results")
        folder = os.path.join(parent_folder,fr_systems_string)
        dataset_folder = os.path.join(folder,train_database.value+"-"+test_dataset.value)
        file_name = os.path.join(dataset_folder,"fmr="+str(test_fmr))
        if os.path.exists(file_name): return Result.load(file_name)    
        print(file_name+" does not exist")
        return None
