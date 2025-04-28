import pandas as pd
import numpy as np

# calculate FP and FN for a threshold
def calculate_fp_fn(df, cluster, threshold):
    df_cluster = df[df['Cluster'] == cluster]
    FP = 0
    FN = 0
    
    for _, row in df_cluster.iterrows():
        score = row['Cosine Distance'] 
        label = row['Label']      # 1 = same person, 0 = different people
        prediction = int(score <= threshold)  # based on threshold
        
        if label == 1 and prediction == 0:  # false negative
            FN += 1
        elif label == 0 and prediction == 1:  # false positive
            FP += 1
    
    return FP, FN


def calculate_eer(df, cluster, possible_thresholds):
    best_threshold = None
    smallest_difference = float('inf')
    eer = None
    
    for threshold in possible_thresholds:
        FP, FN = calculate_fp_fn(df, cluster, threshold)
        FPR = FP / max(1, len(df[df['Label'] == 0]))  # False Positive Rate
        FNR = FN / max(1, len(df[df['Label'] == 1]))  # False Negative Rate
        
        difference = abs(FPR - FNR)
        
        if difference < smallest_difference:
            smallest_difference = difference
            best_threshold = threshold
            eer = (FPR + FNR) / 2  # FPR ≈ FNR
    
    return eer, best_threshold

# example
csv_sp = r'C:\Users\emily\OneDrive\Documents\IC\algorithms\deep face\DifferentPerson_train_dp.csv' 
csv_dp = r'C:\Users\emily\OneDrive\Documents\IC\algorithms\deep face\DifferentPerson_train_dp.csv'

df_sp = pd.read_csv(csv_sp)
df_dp = pd.read_csv(csv_dp)

df_sp['Label'] = 1  # same person
df_dp['Label'] = 0  # different people

df = pd.concat([df_sp, df_dp])

unique_clusters = sorted(df['Cluster'].unique())

# list of possible thresholds
possible_thresholds = np.linspace(0.65, 0.85, 100)  # adjust if needed

# calculate EER and ideal threshold for each cluster
eer_per_cluster = {}
thresh_per_cluster = {}

for cluster in unique_clusters:
    eer, ideal_threshold = calculate_eer(df, cluster, possible_thresholds)
    eer_per_cluster[cluster] = eer
    thresh_per_cluster[cluster] = ideal_threshold
    print(f"Cluster {cluster}: EER = {eer:.4f}, Ideal Threshold = {ideal_threshold:.4f}")
