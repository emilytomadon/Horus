import pandas as pd
import numpy as np

def calculate_fp_fn(df, cluster, threshold):
    df_cluster = df[df['Cluster'] == cluster]
    FP = 0
    FN = 0
    
    for _, row in df_cluster.iterrows():
        score = row['Cosine Distance']
        label = row['Label']
        prediction = int(score <= threshold)
        
        if label == 1 and prediction == 0:
            FN += 1
        elif label == 0 and prediction == 1:
            FP += 1
    
    return FP, FN

def find_best_threshold(df, cluster, possible_thresholds):
    best_threshold = None
    lowest_total_error = float('inf')
    
    for threshold in possible_thresholds:
        FP, FN = calculate_fp_fn(df, cluster, threshold)
        total_error = FP + FN
        
        if total_error < lowest_total_error:
            lowest_total_error = total_error
            best_threshold = threshold
    
    return best_threshold, lowest_total_error

csv_sp = r'C:\Users\emily\OneDrive\Documents\IC\algorithms\deep face\SamePerson_train_dp.csv'
csv_dp = r'C:\Users\emily\OneDrive\Documents\IC\algorithms\deep face\DifferentPerson_train_dp.csv'

df_sp = pd.read_csv(csv_sp)
df_dp = pd.read_csv(csv_dp)

df_sp['Label'] = 1
df_dp['Label'] = 0

df = pd.concat([df_sp, df_dp])

unique_clusters = sorted(df['Cluster'].unique())

possible_thresholds = np.linspace(0.65, 0.85, 100)

thresh_per_cluster = {}
total_error_per_cluster = {}

for cluster in unique_clusters:
    ideal_threshold, lowest_total_error = find_best_threshold(df, cluster, possible_thresholds)
    thresh_per_cluster[cluster] = ideal_threshold
    total_error_per_cluster[cluster] = lowest_total_error
    print(f"Cluster {cluster}: Ideal threshold = {ideal_threshold:.4f}, Total error (FP + FN) = {lowest_total_error}")
