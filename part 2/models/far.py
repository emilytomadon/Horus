import pandas as pd
import numpy as np

def calculate_fp_fn_far_frr(df, cluster, threshold):
    df_cluster = df[df['Cluster'] == cluster]
    FP = 0
    FN = 0
    total_neg_pairs = len(df_cluster[df_cluster['Label'] == 0])
    total_pos_pairs = len(df_cluster[df_cluster['Label'] == 1])
    
    for _, row in df_cluster.iterrows():
        score = row['Cosine Distance']
        label = row['Label']
        prediction = int(score <= threshold)
        
        if label == 1 and prediction == 0:
            FN += 1
        elif label == 0 and prediction == 1:
            FP += 1
    
    FAR = FP / max(1, total_neg_pairs)
    FRR = FN / max(1, total_pos_pairs)
    
    return FP, FN, FAR, FRR

def calculate_threshold_for_far(df, cluster, possible_thresholds, target_far):
    best_threshold = None
    lowest_frr = float('inf')
    corresponding_far = None
    
    for threshold in possible_thresholds:
        FP, FN, FAR, FRR = calculate_fp_fn_far_frr(df, cluster, threshold)
        
        if FAR <= target_far and FRR < lowest_frr:
            lowest_frr = FRR
            best_threshold = threshold
            corresponding_far = FAR
    
    return best_threshold, corresponding_far, lowest_frr

#example
csv_sp = r'C:\Users\emily\OneDrive\Documents\IC\algorithms\deep face\SamePerson_train_dp.csv'
csv_dp = r'C:\Users\emily\OneDrive\Documents\IC\algorithms\deep face\DifferentPerson_train_dp.csv'

df_sp = pd.read_csv(csv_sp)
df_dp = pd.read_csv(csv_dp)

df_sp['Label'] = 1
df_dp['Label'] = 0

df = pd.concat([df_sp, df_dp])

unique_clusters = sorted(df['Cluster'].unique())

possible_thresholds = np.linspace(0.65, 0.85, 100)

target_far = 0.01 # adjust if needed

thresh_per_cluster = {}
far_per_cluster = {}
frr_per_cluster = {}

for cluster in unique_clusters:
    ideal_threshold, corresponding_far, corresponding_frr = calculate_threshold_for_far(
        df, cluster, possible_thresholds, target_far)
    thresh_per_cluster[cluster] = ideal_threshold
    far_per_cluster[cluster] = corresponding_far
    frr_per_cluster[cluster] = corresponding_frr
    print(f"Cluster {cluster}: Ideal threshold = {ideal_threshold:.4f}, FAR = {corresponding_far:.4f}, FRR = {corresponding_frr:.4f}")
