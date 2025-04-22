import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__))))

import os

from matplotlib import pyplot as plt
from FALCON.falcon import FALCON
from helper_classes.dataset import FRDataset
from helper_classes.result import AttributeResult, Result
from tools.enums import Attribute, Dataset, FRSystem, Method, Metric
import pandas as pd
import seaborn as sns

class ParameterExperiment(FALCON):
    def __init__(self,
            train_dataset,
            test_datasets,
            kernel,
            regularization_alpha,
            omega
    ):
        super().__init__(train_dataset,test_datasets,kernel,regularization_alpha, omega=omega, nearest_neighbors = None,train_fmr= 0.001,test_fmrs=[0.001])
    
    def save_results(self, result, test_dataset:FRDataset, fmr):
        CWD = os.path.abspath(os.getcwd())
        parent_folder = os.path.join(os.path.join(CWD, self.method.value),"parameter")
        folder = os.path.join(os.path.join(parent_folder,self._fr_system.value),self._train_dataset.dataset.value+"-"+test_dataset.dataset.value)
        kernel_folder = os.path.join(folder,self._kernel)
        if not os.path.exists(kernel_folder):
                os.makedirs(kernel_folder)
        result.save(os.path.join(kernel_folder, f"alpha={self._regularization_alpha}-omega={self._omega}"))         
    
    @staticmethod
    def get_result_from_file(fr_system : FRSystem, train_database : Dataset, test_dataset: Dataset, kernel, alpha, omega, fmr=0.001):
        CWD = os.path.abspath(os.getcwd())  # Current working director
        if fmr != 0.001 and fmr != 0.0001: raise ValueError("This FMR is not supported")
        parent_folder = os.path.join(os.path.join(CWD, FALCON.method.value),"parameter"+("_0001"if fmr == 0.0001 else ""))     
        fr_folder = os.path.join(parent_folder,fr_system.value)
        dataset_folder = os.path.join(fr_folder,train_database.value+"-"+test_dataset.value)
        file_name = os.path.join(os.path.join(dataset_folder,kernel),"alpha="+str(alpha)+"-omega="+str(omega))
        if os.path.exists(file_name): return Result.load(file_name)    
        print(file_name+" does not exist")
        return None

def falcon(fr_system : FRSystem, train_dataset : Dataset, test_dataset: Dataset, kernel: str, regularization_alpha: float, omega: float):
    falcon = ParameterExperiment(FRDataset(fr_system, train_dataset),[FRDataset(fr_system, test_dataset)],kernel=kernel, regularization_alpha=regularization_alpha, omega = omega)
    falcon.train()
    return falcon.test()



































def gather_data_and_print(fr_system: FRSystem, train_dataset:Dataset, test_dataset:Dataset, kernels:list[str], alphas:list[float], omegas:list[float], fmr = 0.001):
    result = {}
    for kernel in kernels:
        result[kernel] = {}
        for alpha in alphas:
            result[kernel][alpha] = {}
            for omega in omegas:
                result[kernel][alpha][omega] = ParameterExperiment.get_result_from_file(fr_system, train_dataset, test_dataset, kernel, alpha, omega, fmr)
                if result[kernel][alpha][omega] is None:
                    print(f"Calculate for kernel = {kernel}, alpha = {str(alpha)}  and omega = {str(omega)}")
                    single_result = falcon(fr_system, train_dataset, test_dataset, kernel, alpha, omega)
                    assert len(list(single_result.keys())) == 1
                    single_result = list(single_result.values())[0]
                    assert len(list(single_result.keys())) == 1
                    single_result : Result = list(single_result.values())[0]
                    result[kernel][alpha][omega] = single_result
    return result

def create_table(results: dict[dict[dict[Result]]], performance: bool, attribute:Attribute):    
    kernels : list[str] = list(results.keys())
    alphas : list[float] = list(list(results.values())[0].keys())
    omegas : list[float] = list(list(list(results.values())[0].values())[0])
    fairness_metrics = Metric.GARBE

    x = {omega: {(Metric.PERFORMANCE if performance else Metric.GARBE).value: [None] * len(kernels)*len(alphas)} for omega in omegas}

    i = 0
    for kernel in kernels:
        for alpha in alphas:
            for omega, result in results[kernel][alpha].items():
                if performance:
                    x[omega][Metric.PERFORMANCE.value][i] = result.accuracy
                else:
                    fairness_metric: AttributeResult = result.fairness_results[attribute]
                    x[omega][fairness_metrics.value][i]=fairness_metric.get_metric(fairness_metrics)
            i += 1

    table = []
    for omega in x.keys():
        for attribute, values in x[omega].items():
            table.append([omega]+values)
    columns = [(kernel,alpha) for kernel in kernels for alpha in alphas]
    return pd.DataFrame(table, columns=["Omega"]+columns)

def get_data(fr_system, train_dataset, test_dataset, kernels, alphas, omegas, performance: bool, attribute:Attribute = None, fmr = 0.001):
    assert not (performance and attribute != None)
    data = gather_data_and_print(fr_system,train_dataset,test_dataset,kernels,alphas, omegas, fmr)
    df = create_table(data, performance, attribute)
    return df.set_index("Omega")

def multiple_heatmaps(fr_system, train_dataset, test_dataset, attribute, save=False, fmr = 0.001):
    from  matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256)
    trade_offs = [0.0, 0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig, axs = plt.subplots(4, 3,sharey=True, sharex=True,figsize=(8, 6.14))
    cbar_ax = fig.add_axes([.91, .07, .03, .875])
    trade_off_dfs = []
    global_max = -1000
    global_min = 1000
    for i in range(12):
        trade_off_df = get_trade_off_df(fr_system, train_dataset, test_dataset, attribute, trade_offs[i], fmr=fmr)
        trade_off_df = trade_off_df.T
        trade_off_dfs.append(trade_off_df)
        local_max = trade_off_df.max().max()
        # print(local_max)
        if local_max > global_max: global_max = local_max
        local_min = trade_off_df.min().min()
        if local_min < global_min: global_min = local_min

    
    for i, ax in enumerate(axs.flat):
        s = sns.heatmap(trade_off_dfs[i], cbar=i == 0, annot=False, fmt=".2f", cmap=cmap, ax=axs[i//3][i%3],vmin=global_min, vmax=global_max,cbar_ax=None if i else cbar_ax, xticklabels=[], yticklabels=[])
        # s = sns.heatmap(trade_off_dfs[i], cbar=i == 0, annot=False, fmt=".2f", cmap=cmap, ax=axs[i//3][i%3],vmin=-0.5, vmax=0.5,cbar_ax=None if i else cbar_ax, xticklabels=[], yticklabels=[])
        # s.set_title(f"{fr_system.value}, {train_dataset.value}-{test_dataset.value}, {attribute.value}")
        # s.set_ylabel('Kernel - Regularisation Alpha')
        # s.set_xlabel('Omega')
        s.set_title(f"$t={trade_offs[i]}$")
        s.tick_params(left=False, bottom=False)
        s.set_xlabel('')
        s.set_ylabel('')

    fig.supxlabel('$\omega$')
    fig.supylabel('Kernel and Regularisation $\\alpha$')
    

    fig.tight_layout(rect=[0.0, 0.0, 0.9, 1])

    if save: 
        folder =os.path.join(os.path.abspath(os.getcwd()), "Bachelorarbeit","figures","parameter_trade_off")
        if not os.path.exists(folder): os.makedirs(folder)
        plt.savefig(os.path.join(folder,fr_system.value+"-"+train_dataset.value+"-"+test_dataset.value+"-"+attribute.value), bbox_inches='tight',  dpi=500)
    else: plt.show() 

def difference_heatmaps(test_dataset, attribute, save=False):
    from  matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256)
    fig, axs = plt.subplots(2, 4,sharey=True, sharex=True)
    cbar_ax = fig.add_axes([.91, .03, .03, .9])
    trade_off_dfs = []
    global_max = -1000
    global_min = 1000

    fr_systems = [FRSystem.FACENET, FRSystem.ARCFACE, FRSystem.MAGFACE,FRSystem.QMAGFACE]
    datasets = [Dataset.LFW, Dataset.COLORFERET, Dataset.ADIENCE]
    datasets.remove(test_dataset)
    for j in range(len(datasets)):
        train_dataset = datasets[j]
        if train_dataset == test_dataset: continue
        for i in range(len(fr_systems)):
            fr_system = fr_systems[i]
            trade_off_df = get_trade_off_df(fr_system, train_dataset, test_dataset, attribute, 0.33)
            trade_off_df = trade_off_df.T
            trade_off_dfs.append(trade_off_df)
            local_max = trade_off_df.max().max()
            if local_max > global_max: global_max = local_max
            local_min = trade_off_df.min().min()
            if local_min < global_min: global_min = local_min
    
    for i in range(8):
        s = sns.heatmap(trade_off_dfs[i], cbar=i == 0, annot=False, fmt=".2f", cmap=cmap, ax=axs[i//4][i%4],vmin=global_min, vmax=global_max,cbar_ax=None if i else cbar_ax, xticklabels=[], yticklabels=[])
        # s = sns.heatmap(trade_off_dfs[i], cbar=i == 0, annot=False, fmt=".2f", cmap=cmap, ax=axs[i//3][i%3],vmin=-0.5, vmax=0.5,cbar_ax=None if i else cbar_ax, xticklabels=[], yticklabels=[])
        if i < 4:
            s.set_title(fr_systems[i].value)
        if i == 0:
            s.set_ylabel(datasets[0].value)
        elif i == 4:
            s.set_ylabel(datasets[1].value)
        else:
            s.set_ylabel('')
        s.tick_params(left=False, bottom=False)
        s.set_xlabel('')
        

    # fig.supxlabel('$\omega$')
    # fig.supylabel('Kernel and Regularisation $\\alpha$')
    

    fig.tight_layout(rect=[0.0, 0.0, 0.9, 1])

    if save: 
        folder =os.path.join(os.path.abspath(os.getcwd()), "Bachelorarbeit","figures","parameter_multi_setting")
        if not os.path.exists(folder): os.makedirs(folder)
        plt.savefig(os.path.join(folder,test_dataset.value+"-"+attribute.value), bbox_inches='tight',  dpi=500)
    else: plt.show() 


def common_heatmap(fr_system, train_dataset, test_dataset, attribute, trade_off, save=False):
    plt.figure(figsize=(8, 6.14))
    trade_off_df = get_trade_off_df(fr_system, train_dataset, test_dataset, attribute, trade_off)

    trade_off_df = trade_off_df.rename(columns=lambda x: str(x)[1:-1])    
    trade_off_df = trade_off_df.rename(columns=lambda x: x.replace("'",""))
    trade_off_df = trade_off_df.rename(columns=lambda x: x.replace("mial","mial "))
    trade_off_df.columns = trade_off_df.columns.str.capitalize()
    trade_off_df = trade_off_df.rename(columns=lambda x: 'RBF' + x[3:] if x.startswith("Rbf") else x)

    trade_off_df = trade_off_df.T
    from  matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256)
    
    s = sns.heatmap(trade_off_df, annot=False, fmt=".2f", cmap=cmap)
    s.set_ylabel('Kernel and Regularisation $\\alpha$')
    s.set_xlabel('$\omega$')
    plt.tight_layout(rect=[0.0, 0, 1, 1])

    if save: 
        folder =os.path.join(os.path.abspath(os.getcwd()), "Bachelorarbeit","figures","parameter")
        if not os.path.exists(folder): os.makedirs(folder)
        plt.savefig(os.path.join(folder,fr_system.value+"-"+train_dataset.value+"-"+test_dataset.value+"-"+attribute.value), bbox_inches='tight', dpi=500)
    else: plt.show() 

def get_trade_off_df(fr_system, train_dataset, test_dataset, attribute, trade_off, return_all = False, kernels = None, alphas = None, omegas = None, fmr = 0.001):
    kernels = kernels if kernels != None else ['linear', 'polynomial2', 'polynomial3', 'rbf'] 
    alphas = alphas if alphas != None else [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] #[0.000001, 0.00001,0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    omegas = omegas if omegas != None else [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9]
    
    from main import cross_ds_validation
    result = cross_ds_validation(Method.BASELINE, fr_system, train_dataset, [test_dataset], attribute = attribute)
    result:Result = list(list(result.values())[0].values())[0]
    baseline_acc = result.accuracy
    baseline_fair = result.fairness_results[attribute].get_metric(Metric.GARBE)

    df_acc = get_data(fr_system, train_dataset, test_dataset, kernels, alphas, omegas, True, fmr = fmr)
    df_acc_rel = df_acc/baseline_acc - 1
    df_fair = get_data(fr_system, train_dataset, test_dataset, kernels, alphas, omegas, False, attribute, fmr = fmr)
    df_fair_ref = df_fair/baseline_fair - 1
    trade_off_df = trade_off * df_acc_rel + (1-trade_off) * df_fair_ref
    if return_all: return df_acc, df_fair, df_acc_rel, df_fair_ref, trade_off_df
    return trade_off_df

def get_best_values(fr_system, train_dataset, test_dataset, attribute, trade_off, relative = False, combination = False, kernels = None, alphas = None, omegas = None, fmr = 0.001):
    df_acc, df_fair, df_acc_rel, df_fair_ref, trade_off_df = get_trade_off_df(fr_system, train_dataset, test_dataset, attribute, trade_off, return_all=True, kernels=kernels, alphas=alphas, omegas=omegas, fmr=fmr)

    min_index = trade_off_df.values.argmin()
    row_index, col_index = divmod(min_index, trade_off_df.shape[1])

    if combination: 
        if relative: return df_acc_rel.iloc[row_index, col_index], df_fair_ref.iloc[row_index, col_index], trade_off_df.index[row_index], trade_off_df.columns[col_index]
        else: return df_acc.iloc[row_index, col_index], df_fair.iloc[row_index, col_index], trade_off_df.index[row_index], trade_off_df.columns[col_index]
    else:
        if relative: return df_acc_rel.iloc[row_index, col_index], df_fair_ref.iloc[row_index, col_index]
        else: return df_acc.iloc[row_index, col_index], df_fair.iloc[row_index, col_index]

def get_best_generalized_value(fr_system, train_dataset, trade_off, combination = False, fmr = 0.001):
    kernels = ['linear', 'polynomial2', 'polynomial3', 'rbf'] 
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    omegas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9]

    datasets = [Dataset.ADIENCE, Dataset.COLORFERET, Dataset.LFW]
    df_rels = []
    df_fair_rel_mean = None
    for test_dataset in datasets:
        if train_dataset == test_dataset: continue
        from main import cross_ds_validation
        baseline_result = cross_ds_validation(Method.BASELINE, fr_system, train_dataset, [test_dataset])
        baseline_result:Result = list(list(baseline_result.values())[0].values())[0]
        baseline_acc = baseline_result.accuracy
        attributes = list(FRDataset(fr_system, test_dataset).subgroups.keys())
        df_acc = get_data(fr_system, train_dataset, test_dataset, kernels, alphas, omegas, True, fmr = fmr)
        df_acc_rel = df_acc/baseline_acc - 1
        df_fair_rels = []
        for attribute in attributes:
            baseline_fair = baseline_result.fairness_results[attribute].get_metric(Metric.GARBE)
            df_fair = get_data(fr_system, train_dataset, test_dataset, kernels, alphas, omegas, False, attribute, fmr = fmr)
            df_fair_rels.append(df_fair/baseline_fair - 1)        
        
        df_fair_rel_mean = pd.concat(df_fair_rels).groupby(level=0).mean()
        df_rels.append(trade_off * df_acc_rel + (1-trade_off) * df_fair_rel_mean)
    df_rel_mean = pd.concat(df_rels).groupby(level=0).mean()

    min_index = df_rel_mean.values.argmin()
    row_index, col_index = divmod(min_index, df_rel_mean.shape[1])
    if combination: 
        return df_acc.iloc[row_index, col_index], df_fair.iloc[row_index, col_index], df_rel_mean.index[row_index], df_rel_mean.columns[col_index]
    return df_rel_mean.index[row_index], df_rel_mean.columns[col_index]

def get_best_generalized_results(fr_system, train_dataset, trade_off, fmr = 0.001):
    parameters = get_best_generalized_value(fr_system, train_dataset, trade_off, combination = True, fmr = fmr)
    omega = parameters[2]
    kernel = parameters[3][0]
    alpha = parameters[3][1]

    return omega, kernel, alpha

def mass_training():
    fr_system = FRSystem.FACENET
    train_dataset = Dataset.ADIENCE
    kernels = ['linear', 'polynomial2', 'polynomial3', 'rbf'] 
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    omegas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # test_dataset = Dataset.COLORFERET

    for kernel in kernels:
        for alpha in alphas:
            for omega in omegas:
                fairness_approach  = FALCON(FRDataset(fr_system, train_dataset),[FRDataset(fr_system, Dataset.COLORFERET),FRDataset(fr_system, Dataset.LFW)], test_fmrs=[0.001], train_fmr=0.001,omega = omega, kernel=kernel, regularization_alpha=alpha)
                fairness_approach.train()
                calculated_results = fairness_approach.test()


if __name__ == "__main__":
    # mass_training()
    fr_system = FRSystem.ARCFACE
    train_dataset = Dataset.LFW
    test_dataset = Dataset.ADIENCE
    attribute = Attribute.GENDER

    # multiple_heatmaps(fr_system,train_dataset,test_dataset, attribute,save=True)
    # common_heatmap(fr_system,train_dataset,test_dataset, attribute, trade_off=0.33, save=True)
    

    # get_best_generalized_results(fr_system, train_dataset, None, None, None, 0.33)

    # for k in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    # common_heatmap(fr_system,train_dataset, test_dataset, attribute, trade_off=0.5)
    
    # datasets = [Dataset.LFW, Dataset.COLORFERET, Dataset.ADIENCE]
    # for fr_system in [FRSystem.FACENET, FRSystem.ARCFACE, FRSystem.MAGFACE,FRSystem.QMAGFACE]:
    #     for i in range(len(datasets)):
    #         for j in range(len(datasets)):
    #             if i == j:continue
    #             for attribute in FRDataset(fr_system,datasets[j]).subgroups.keys():
    #                 multiple_heatmaps(fr_system,datasets[i], datasets[j], attribute,save=True)
    #                 common_heatmap(fr_system,datasets[i], datasets[j], attribute, trade_off=0.33, save=True)


    # test_dataset = Dataset.COLORFERET
    # attribute = Attribute.ETHNICS
    # difference_heatmaps(test_dataset, attribute)

    for dataset in [Dataset.ADIENCE, Dataset.COLORFERET, Dataset.LFW]:
        for attribute in FRDataset(FRSystem.ARCFACE,dataset).subgroups.keys():
            difference_heatmaps(dataset, attribute, save=False)


    # for k in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    #     print(get_best_values(fr_system, train_dataset, test_dataset, attribute,k, combination=True))


    # common_df = pd.concat(dfs, axis=1)
    # common_df = common_df.groupby(common_df.columns, axis=1).mean()
    # datasets = [Dataset.ADIENCE, Dataset.COLORFERET, Dataset.LFW]
    # for fr_system in [FRSystem.ARCFACE, FRSystem.FACENET, FRSystem.MAGFACE]:#, FRSystem.QMAGFACE]:
    #     for i in range(len(datasets)):
    #         train_dataset = datasets[i]
    #         for j in range(len(datasets)):
    #             if j == i: continue
    #             test_dataset = datasets[j]
    #             for attribute in list(FRDataset(fr_system, test_dataset).subgroups.keys()):
    #                 common_heatmap(fr_system,train_dataset, test_dataset, attribute, trade_off=0.5, save=True)

    # df = gather_data_and_print(fr_system,train_dataset,test_dataset,kernels, alphas)
    # print(df)
