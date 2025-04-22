'''
MIT License

Copyright (c) 2025 Philipp Hempel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
from FSN.fsn import FSN
from FairCal.fair_cal import FairCal
from FTC.ftc import FTC
from Baseline.baseline import Baseline
from FALCON.falcon import FALCON
from helper_classes.dataset import FRDataset
from helper_classes.fairness_approach import Fairness_Approach
from SLF.slf import SLF
from tools.enums import Attribute, Dataset, FRSystem, Method    

attribute_abbs = {
    "A":Attribute.AGE,
    "E":Attribute.ETHNICS,
    "G":Attribute.GENDER
}

def cross_ds_validation(method: Method, fr_system : FRSystem, train_database : Dataset, test_datasets: list[Dataset], test_fmrs: list[float] = [0.001], recalculate = False, train_fmr:float = 0.001, attribute:Attribute = None):
    test_datasets_to_calculate = test_datasets.copy()
    results = {}
    if not recalculate:
        for dataset in test_datasets:
            results[dataset] = {}
            got_all_files = True
            for fmr in test_fmrs:
                result = get_result_from_file(method, fr_system, train_database, dataset, train_fmr, fmr, attribute)#get_file_results(method, fr_system , train_database , dataset, fmr)
                if result is None: 
                    got_all_files = False
                    del results[dataset]
                    break
                else: results[dataset][fmr] = result
            if got_all_files: test_datasets_to_calculate.remove(dataset)
    if len(test_datasets_to_calculate) > 0:
        fairness_approach : Fairness_Approach = get_fairness_approach(method, fr_system, train_database, test_datasets_to_calculate, train_fmr, test_fmrs, attribute)
        fairness_approach.train()
        calculated_results = fairness_approach.test()
        results.update(calculated_results)
    return results

def get_one_result(method: Method, fr_system : FRSystem, train_database : Dataset, test_dataset: Dataset, test_fmr: float = 0.001, recalculate = False):
    x = cross_ds_validation(method, fr_system, train_database, [test_dataset], [test_fmr], recalculate)
    return x[test_dataset][test_fmr]

def get_result_from_file(method,fr_system, train_database, dataset, train_fmr, test_fmr, attribute):
    if method == Method.BASELINE: return Baseline.get_result_from_file(fr_system, dataset, test_fmr)
    if method == Method.FSN: return FSN.get_result_from_file(fr_system, train_database, dataset, train_fmr, test_fmr)
    if method == Method.SLF: 
        fr_systems = [fr_system, FRSystem.ARCFACE if fr_system != FRSystem.ARCFACE else FRSystem.FACENET] # Arcface is set as second FRSystem every time except Arcface is given, then FaceNet
        return SLF.get_result_from_file(fr_systems, train_database, dataset, test_fmr)
    if method == Method.FTC and attribute != None:  return FTC.get_result_from_file(fr_system, train_database, dataset, test_fmr, attribute)
    if isinstance(method,str) and method.startswith(Method.FTC.value): return FTC.get_result_from_file(fr_system, train_database, dataset, test_fmr, attribute_abbs[method[-1]])
    if method == Method.FAIRCAL: return FairCal.get_result_from_file(fr_system,train_database,dataset,test_fmr)
    if method == Method.FALCON: 
        if True:
            from experiments.parameter_experiment import get_best_generalized_results
            omega, kernel, alpha = get_best_generalized_results(fr_system,train_database, trade_off=0.33)
            
            return FALCON.get_result_from_file(fr_system, train_database, dataset, train_fmr, test_fmr, kernel = kernel, alpha = alpha, omega = omega)
    raise ValueError("No available method selected: ",method)

def baseline(fr_system : FRSystem, test_datasets: list[Dataset], test_fmrs: list[float] = [0.001]):
    return Baseline([FRDataset(fr_system, ds) for ds in test_datasets], test_fmrs=test_fmrs)

def fsn(fr_system : FRSystem, train_dataset : Dataset, test_datasets: list[Dataset], train_fmr:float = 0.001, test_fmrs: list[float] = [0.001]):
    train_dataset = FRDataset(fr_system, train_dataset)
    train_dataset = train_dataset.get_subset(train_dataset.not_singleton_mask())
    return FSN(train_dataset,[FRDataset(fr_system, ds) for ds in test_datasets], test_fmrs=test_fmrs, train_fmr=train_fmr)

def slf(fr_systems: list[FRSystem], train_dataset : Dataset, test_datasets : list[Dataset], test_fmrs: list[float] = [0.001]):
    return SLF(fr_systems=fr_systems,test_databases=test_datasets, train_database=train_dataset, test_fmrs=test_fmrs)

def ftc(fr_system: FRSystem, train_dataset : Dataset, test_datasets : list[Dataset], test_fmrs: list[float] = [0.001], attribute: Attribute = None, sampling = 0.5):
    train_frdataset = FRDataset(fr_system, train_dataset)
    mask = np.ones(len(train_frdataset),dtype=bool)
    mask[np.random.choice(range(len(train_frdataset)),round(np.sum(len(train_frdataset)*(sampling))),replace=False)] = False
    train_frdataset = train_frdataset.get_subset(mask)

    test_datasets = [FRDataset(fr_system, ds) for ds in test_datasets]
    for i in range(len(test_datasets)):
        mask = np.ones(len(test_datasets[i]),dtype=bool)
        mask[np.random.choice(range(len(test_datasets[i])),round(np.sum(len(test_datasets[i])*(sampling))),replace=False)] = False
        test_datasets[i] = test_datasets[i].get_subset(mask)

    return FTC(train_frdataset,test_datasets, attribute,test_fmrs)

def fair_cal(fr_system : FRSystem, train_dataset : Dataset, test_datasets: list[Dataset], test_fmrs: list[float] = [0.001]):
    train_dataset = FRDataset(fr_system, train_dataset)
    train_dataset = train_dataset.get_subset(train_dataset.not_singleton_mask())
    return FairCal(train_dataset,[FRDataset(fr_system, ds) for ds in test_datasets], test_fmrs=test_fmrs)

def falcon(fr_system : FRSystem, train_dataset : Dataset, test_datasets: list[Dataset], train_fmr:float = 0.001, test_fmrs: list[float] = [0.001]):
    return FALCON(FRDataset(fr_system, train_dataset),[FRDataset(fr_system, ds) for ds in test_datasets], test_fmrs=test_fmrs, train_fmr=train_fmr)

def get_fairness_approach(method: Method, fr_system : FRSystem, train_dataset : Dataset, test_datasets: list[Dataset], train_fmr:float = 0.001,test_fmrs: list[float] = [0.001], attribute: Attribute = None):
    if method == Method.BASELINE: return baseline(fr_system,test_datasets,test_fmrs)
    if method == Method.FSN: return fsn(fr_system,train_dataset,test_datasets,train_fmr, test_fmrs)
    if method == Method.SLF: 
        fr_systems = [fr_system, FRSystem.ARCFACE if fr_system != FRSystem.ARCFACE else FRSystem.FACENET]
        return slf(fr_systems,train_dataset,test_datasets,test_fmrs)
    if method == Method.FTC and attribute != None:  return ftc(fr_system,train_dataset,test_datasets, test_fmrs, attribute)
    if isinstance(method,str) and method.startswith(Method.FTC.value): return ftc(fr_system,train_dataset,test_datasets, test_fmrs, attribute_abbs[method[-1]])
    if method == Method.FAIRCAL: return fair_cal(fr_system,train_dataset,test_datasets,test_fmrs)
    if method == Method.FALCON: return falcon(fr_system,train_dataset,test_datasets,train_fmr, test_fmrs)
    raise ValueError("No available method selected")

if __name__ == "__main__":
    fr_system = FRSystem.ARCFACE
    train_dataset = Dataset.RFW_INDIAN
    test_dataset = Dataset.RFW_VAL_INDIAN
    attribute = Attribute.ETHNICS

    for method in [Method.FALCON]: #Method:
        print(method.value)
        approach = get_fairness_approach(method, fr_system, train_dataset, [test_dataset],attribute= 
        attribute)
        approach.train()
        approach.test()