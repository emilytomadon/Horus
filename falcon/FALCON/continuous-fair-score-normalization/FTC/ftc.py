

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__))))

from helper_classes.fairness_approach import Fairness_Approach
import math
from tqdm import tqdm
from helper_classes.dataset import FRDataset
from FairCal.beta_cal import BetaCalibration
import numpy as np
from sklearn.metrics import roc_curve

import numpy as np
import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from helper_classes.result import Result
from tools.enums import Attribute, FRSystem, Method


class FTC(Fairness_Approach):
    method : Method = Method.FTC
    def __init__(self, 
        train_dataset: FRDataset, 
        test_datasets: list[FRDataset],
        attribute: Attribute,
        test_fmrs = [0.001],
        sampling = 1.0,
        gen_per_imp_comp = 1
        ):        
        super().__init__(train_dataset,test_datasets, test_fmrs = test_fmrs)
        self._attribute = attribute
        self._model = None
        self._calibration = None
        self._loss_fn = None
        self._sampling = sampling
        self._gen_per_imp_comp = gen_per_imp_comp

    def get_samples(self, dataset: FRDataset):
        ids_matrix = np.array([dataset.ids])
        genuine_indices_matrix = np.equal(ids_matrix,np.transpose(ids_matrix))
        genuine_indices_triangle = np.bitwise_and(genuine_indices_matrix, np.triu(np.ones((len(dataset), len(dataset)), dtype=bool), 1))

        all_genuine_indices = np.column_stack(np.where(genuine_indices_triangle))
        sampled_genuine_indices = all_genuine_indices[np.random.choice(range(len(all_genuine_indices)),size=math.ceil(len(all_genuine_indices)*self._sampling),replace=False)]

        x = np.random.randint(2, size=sampled_genuine_indices.shape[0])
        imposter_base_indices = sampled_genuine_indices[np.arange(sampled_genuine_indices.shape[0]), x]
        sampled_imposter_indices = None
        imposter_indices_matrix = np.invert(genuine_indices_matrix)
        for index, counts in tqdm(np.column_stack(np.unique(imposter_base_indices, return_counts=True))):
            imposter_indice = np.where(imposter_indices_matrix[index])[0]
            number_imposter_comparisons = min(len(imposter_indice),counts*self._gen_per_imp_comp)
            tuple = list(zip([index]*number_imposter_comparisons,np.random.choice(imposter_indice,size=number_imposter_comparisons, replace=False)))
            if sampled_imposter_indices is None: sampled_imposter_indices = np.array(tuple)
            else: sampled_imposter_indices = np.vstack([sampled_imposter_indices,tuple])
        return sampled_genuine_indices, sampled_imposter_indices

    def train(self):        
        sampled_genuine_indices, sampled_imposter_indices = self.get_samples(self._train_dataset)

        error_embeddings, train_ground_truth, subgroups_left, subgroups_right, _ = self.get_error_embeddings(self._train_dataset, sampled_genuine_indices, sampled_imposter_indices)
        print(error_embeddings,"\n", train_ground_truth, "\n",subgroups_left, "\n",subgroups_right,"\n",_)

        # error_embeddings, train_ground_truth, subgroups_left, subgroups_right, _ = get_error_embeddings(self._train_dataset, self._attribute)
        train_dataloader = DataLoader(
            EmbeddingsDataset(error_embeddings, train_ground_truth, subgroups_left, subgroups_right),
            batch_size=200,
            shuffle=True,
            num_workers=0)
    
        evaluate_train_dataloader = DataLoader(
            EmbeddingsDataset(error_embeddings, train_ground_truth, subgroups_left, subgroups_right),
            batch_size=200,
            shuffle=False,
            num_workers=0)
        
        # Initialize model
        self._model = NeuralNetwork(self._train_dataset.fr_system)
        # Initialize the loss function
        self._loss_fn = nn.CrossEntropyLoss()
        # Initialize optimizer
        optimizer = optim.Adam(self._model.parameters(), lr=1e-6, weight_decay=1e-3) #lr = learning rate 10^-3, 4, ... andere learning rates
        epochs = 50
        for t in tqdm(range(epochs)): 
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop(train_dataloader, self._model, self._loss_fn, optimizer, self._train_dataset, self._attribute)
            #_, _ = test_loop(evaluate_test_dataloader, model, loss_fn)
        print("Done!")

        scores_cal, ground_truth_cal = self.test_loop(evaluate_train_dataloader, self._model, self._loss_fn)
        scores_cal = scores_cal[:, 1].numpy().reshape(-1)
        assert sum(np.array(ground_truth_cal == 1) != np.array(train_ground_truth)) == 0
        self._calibration = BetaCalibration(scores_cal, ground_truth_cal, score_min=-1, score_max=1)
    
    def test(self):
        metrics = {}
        for dataset in self._test_datasets:
            if self._attribute not in list(dataset.subgroups.keys()): continue
            sampled_genuine_indices, sampled_imposter_indices = self.get_samples(dataset)

            error_embeddings, test_ground_truth, subgroups_left, subgroups_right, indices = self.get_error_embeddings(dataset, sampled_genuine_indices, sampled_imposter_indices, subgroups=True)

            evaluate_test_dataloader = DataLoader(
                EmbeddingsDataset(error_embeddings, test_ground_truth, subgroups_left, subgroups_right),
                batch_size=200,
                shuffle=False,
                num_workers=0)      

            fair_scores, ground_truth = self.test_loop(evaluate_test_dataloader, self._model, self._loss_fn)
            fair_scores = fair_scores[:, 1].numpy().reshape(-1) #Necessary??
            assert sum(np.array(ground_truth == 1) != np.array(test_ground_truth)) == 0
            
            confidences = self._calibration.predict(fair_scores)
            ground_truth_numpy = ground_truth.numpy()
            assert np.array_equal(ground_truth_numpy, test_ground_truth)

            
            subgroup_masks = self.get_subgroup_masks(dataset,indices)
            metrics[dataset.dataset] = self.evaluate(dataset,test_ground_truth,confidences,subgroup_masks)
        return metrics
    
    def get_error_embeddings(self,dataset, sampled_genuine_indices, sampled_imposter_indices, subgroups = True):
        embeddings = dataset.embeddings.astype(np.float32)
        results = np.abs(np.vstack([embeddings[sampled_genuine_indices[:,0]]-embeddings[sampled_genuine_indices[:,1]], embeddings[sampled_imposter_indices[:,0]]-embeddings[sampled_imposter_indices[:,1]]]))
        labels = np.concatenate([np.ones(sampled_genuine_indices.shape[0], dtype=bool), np.zeros(sampled_imposter_indices.shape[0], dtype=bool)])
        indices = np.vstack([sampled_genuine_indices, sampled_imposter_indices])
        if subgroups:
            return results, labels, dataset.features[self._attribute][indices[:,0]],  dataset.features[self._attribute][indices[:,1]], indices
        else:
            return results, labels, indices
    
    def fair_individual_loss(self, g1, g2, y, yhat, dataset:FRDataset, attribute: Attribute):
        subgroups = dataset.subgroups[attribute]
        loss = 0
        for i in subgroups:
            for j in subgroups:
                select_i = np.logical_and(np.array(g1) == i, np.array(g2) == i)
                select_j = np.logical_and(np.array(g1) == j, np.array(g2) == j)
                if (sum(select_i) > 0) and (sum(select_j) > 0):
                    select = y[select_i].reshape(-1, 1) == y[select_j]
                    aux = torch.cdist(yhat[select_i, :], yhat[select_j, :])[select].pow(2).sum()
                    loss += aux/(sum(select_i)*sum(select_j))
        return loss


    def train_loop(self, dataloader:DataLoader, model, loss_fn, optimizer, dataset:FRDataset, attribute:Attribute):

        batch_check = 500
        model.cuda()
        size = len(dataloader.dataset)
        for batch, (X, g1, g2, y) in list(enumerate(dataloader)):
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            # Compute prediction and loss
            pred, prob = model(X)
            loss = 0.5*loss_fn(pred, y)+0.5*self.fair_individual_loss(g1, g2, y, pred, dataset,attribute)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % batch_check == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        model.cpu()


    def test_loop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        test_loss, correct = 0, 0

        scores = torch.zeros(0, 2)
        ground_truth = torch.zeros(0)
        with torch.no_grad():
            for X, g1, g2, y in dataloader:
                pred, prob = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                scores = torch.cat((scores, prob))
                ground_truth = torch.cat([ground_truth, y], 0)
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        fpr, tpr, thr = roc_curve(ground_truth, scores[:, 1].numpy())
        print('FNR @ 0.1 FPR %1.2f'% (1-tpr[np.argmin(np.abs(fpr-1e-3))]))
        return scores, ground_truth

    def save_results(self, metrics, test_dataset:FRDataset, fmr):
        CWD = os.path.abspath(os.getcwd())
        parent_folder = os.path.join(os.path.join(CWD, self.method.value),"results")
        folder = os.path.join(os.path.join(parent_folder,self._fr_system.value),self._train_dataset.dataset.value+"-"+test_dataset.dataset.value)
        if not os.path.exists(folder):
                os.makedirs(folder)
        metrics.save(os.path.join(folder, f"fmr={fmr}, attribute={self._attribute.value}"))

    @staticmethod
    def get_result_from_file(fr_system : FRSystem, train_database : Dataset, test_dataset: Dataset, test_fmr: float = 0.001, attribute:Attribute = None):
        CWD = os.path.abspath(os.getcwd())  # Current working directory
        parent_folder = os.path.join(os.path.join(CWD, FTC.method.value),"results")     
        fr_folder = os.path.join(parent_folder,fr_system.value)
        dataset_folder = os.path.join(fr_folder,train_database.value+"-"+test_dataset.value)
        file_name = os.path.join(dataset_folder,"fmr="+str(test_fmr)+", attribute="+attribute.value)
        if os.path.exists(file_name): return Result.load(file_name)    
        print(file_name+" does not exist")
        return None

class NeuralNetwork(nn.Module):
    def __init__(self, fr_system):
        
        # If the embedding width is 512, we have to multiply the in and outputs of the NN with 4
        factor = 4
        if fr_system == FRSystem.FACENET: factor = 1

        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128 * factor, 256 * factor),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256 * factor, 512 * factor),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512 * factor, 512 * factor),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512 * factor, 2)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        prob = self.softmax(logits)
        return logits, prob


class EmbeddingsDataset(Dataset):
    """Embeddings dataset."""

    def __init__(self, error_embeddings, ground_truth, subgroups_left, subgroups_right):
        """
        Arguments
        """
        self.subgroups_left = subgroups_left
        self.subgroups_right = subgroups_right
        self.error_embeddings = error_embeddings
        self.labels = torch.zeros(len(error_embeddings)).type(torch.LongTensor)
        self.labels[ground_truth] = 1

    def __len__(self):
        return len(self.error_embeddings)

    def __getitem__(self, idx):
        if self.subgroups_left is None: return self.error_embeddings[idx], self.labels[idx]
        else: return self.error_embeddings[idx], self.subgroups_left[idx], self.subgroups_right[idx], self.labels[idx]


