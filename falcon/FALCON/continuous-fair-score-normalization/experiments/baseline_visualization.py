import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__))))

import os
from matplotlib import pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.manifold import TSNE
from tqdm import tqdm
from FALCON.falcon import FALCON
from helper_classes.dataset import FRDataset
from tools.fnmr_fmr import fnmr_at_fmr
from tools.enums import Dataset, FRSystem

from sklearn.metrics import pairwise

def draw_plot(position, thresholds,max_thr,min_thr):
    plt.figure(figsize=(8, 4))
    plt.scatter(position[:,0],position[:,1],c=thresholds,cmap='inferno_r',s=5, label='Threshold',vmin=max_thr, vmax=min_thr)
    cbar = plt.colorbar()
    cbar.set_label("Optimal Threshold", rotation=90)
    plt.clim(min_thr, max_thr)
    
    plt.xlabel(r'$x$',size = 14,math_fontfamily='cm')
    plt.ylabel(r'$y$',size = 14,math_fontfamily='cm')
    plt.figtext(0.5, 0, f"Min: {np.round(np.min(thresholds),4)}, Max: {np.round(np.max(thresholds),4)}", wrap=True, horizontalalignment='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.01, 1, 1])
    return plt

def get_tsne(dataset: FRDataset, recalculate = False):
    folder = os.path.join(os.path.abspath(os.getcwd()),"thresholds-visualisations")
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_name = os.path.join(folder,"TSNE-"+dataset.dataset_name+".npy")
    if (not recalculate) and os.path.exists(file_name):
        return np.load(file_name)            
    print(file_name+" does not exist")
    tsne = TSNE(n_components=2, random_state=42)
    data = tsne.fit_transform(dataset.embeddings)  
    
    np.save(file_name,data)    
    return data

def visualization(fr_system, dataset, train_dataset = None, show = True, save = True, recalculate = False):
    fr_dataset = FRDataset(fr_system,dataset)

    fr_dataset = fr_dataset.get_subset(fr_dataset.not_singleton_mask())
    thresholds = local_thresholds_baseline(fr_dataset, recalculate=recalculate)

    assert fr_dataset.embeddings.shape[0] == len(thresholds)

    points = get_tsne(fr_dataset, recalculate=recalculate)
    max_thr = thresholds.max()
    min_thr = thresholds.min()

    

    print(train_dataset)
    if train_dataset != None:
        train_fr_dataset = FRDataset(fr_system,train_dataset)
        thresholds2 = local_thresholds_falcon(train_fr_dataset,fr_dataset, recalculate=recalculate)
        max_thr2 = thresholds2.max()
        min_thr2 = thresholds2.min()
        if max_thr2 > max_thr: max_thr = max_thr2
        if min_thr2 < min_thr: min_thr = min_thr2

    plt = draw_plot(points, thresholds,max_thr,min_thr)
    if save: 
        folder = os.path.join(os.path.abspath(os.getcwd()), "thresholds-visualisations","figures")
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(os.path.join(folder,fr_dataset.dataset_name), bbox_inches='tight')
    
    if train_dataset != None:
        plt = draw_plot(points, thresholds2,max_thr,min_thr)
        if save: 
            folder = os.path.join(os.path.abspath(os.getcwd()), "thresholds-visualisations","figures")
            plt.savefig(os.path.join(folder,train_fr_dataset.dataset_name+"-"+fr_dataset.dataset_name), bbox_inches='tight')
        if show: plt.show()


def cal_optimal_thresholds(ids, scores):
    opt_threshold_per_embedding = np.array([], dtype=float)
    for index in tqdm(range(len(ids)), desc="Local Thresholds"):
        labels = ids == ids[index]
        opt_threshold_per_embedding = np.append(opt_threshold_per_embedding,fnmr_at_fmr(labels,scores[index],return_fpr_thr = True,fmr=0.001)[2])
    assert opt_threshold_per_embedding.size == len(ids), "The thresholds do not math the embeddings array"
    return opt_threshold_per_embedding


# Baseline

def local_thresholds_baseline(dataset: FRDataset, recalculate = False):
        folder = os.path.join(os.path.abspath(os.getcwd()), "thresholds-visualisations")
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_name = os.path.join(folder,dataset.dataset_name+".npy")
        if (not recalculate) and os.path.exists(file_name):
            return np.load(file_name)

        # Calculate similarity scores
        cosine_similarities = metrics.pairwise.cosine_similarity(dataset.embeddings) if dataset.fr_system != FRSystem.QMAGFACE else dataset.qMagFace_comparison_scores()

        # Calculate optimal thresholds for each embedding
        opt_threshold_per_embedding = cal_optimal_thresholds(dataset.ids, cosine_similarities)

        np.save(file_name,opt_threshold_per_embedding)    
        return opt_threshold_per_embedding

# FALCON

class VisualizeNormalization(FALCON):
    def __init__(self, 
        train_dataset: FRDataset, 
        test_dataset: FRDataset, 
        ):
        super().__init__(train_dataset, [test_dataset])

    def normalized_scores(self, unnormed_scores_matrix, indices, dataset : FRDataset, ):
        thresholds = self._krr.predict(dataset.embeddings)
        return self._omega*unnormed_scores_matrix + (1-self._omega)*(1-(thresholds[indices[:,0]]+thresholds[indices[:,1]]))
        
    def test(self):
        test_dataset = self._test_datasets[0]
        cosine_similarities = pairwise.cosine_similarity(test_dataset.embeddings) if test_dataset.fr_system != FRSystem.QMAGFACE else test_dataset.qMagFace_comparison_scores()
        # Normalize:
        thresholds = self._krr.predict(test_dataset.embeddings)
        normalized_scores = self._omega * cosine_similarities + (1-self._omega)*(1-(thresholds + thresholds[:,np.newaxis]))
        return normalized_scores


def local_thresholds_falcon(train_dataset: FRDataset, test_dataset: FRDataset, recalculate = False):
        folder = os.path.join(os.path.abspath(os.getcwd()), "thresholds-visualisations")
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_name = os.path.join(folder,train_dataset.dataset_name+"-"+test_dataset.dataset_name+".npy")
        if (not recalculate) and os.path.exists(file_name):
            return np.load(file_name)

        # Calculate similarity scores
        train_dataset.get_subset(train_dataset.not_singleton_mask())
        vfalcon = VisualizeNormalization(train_dataset,test_dataset)
        vfalcon.train()
        normalized_scores = vfalcon.test()
        
        # Calculate optimal thresholds for each embedding
        opt_threshold_per_embedding = cal_optimal_thresholds(test_dataset.ids, normalized_scores)
        
        np.save(file_name,opt_threshold_per_embedding)    
        return opt_threshold_per_embedding



if __name__ == "__main__":
    fr_system = FRSystem.ARCFACE
    dataset = Dataset.ADIENCE
    visualization(fr_system, Dataset.ADIENCE, train_dataset=Dataset.COLORFERET, save=False, show=True, recalculate=False)