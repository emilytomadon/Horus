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

def find_best_threshold_with_far(df, cluster, possible_thresholds, target_far):
    best_threshold = None
    lowest_total_error = float('inf')
    best_far = None
    best_frr = None

    for threshold in possible_thresholds:
        FP, FN, FAR, FRR = calculate_fp_fn_far_frr(df, cluster, threshold)
        total_error = FP + FN

        if FAR <= target_far and total_error < lowest_total_error:
            lowest_total_error = total_error
            best_threshold = threshold
            best_far = FAR
            best_frr = FRR

    # Se nenhum threshold atender ao target_far, pega o menor FAR possível
    if best_threshold is None:
        for threshold in possible_thresholds:
            FP, FN, FAR, FRR = calculate_fp_fn_far_frr(df, cluster, threshold)
            total_error = FP + FN
            if best_far is None or FAR < best_far or (FAR == best_far and total_error < lowest_total_error):
                best_far = FAR
                best_frr = FRR
                best_threshold = threshold
                lowest_total_error = total_error

    return best_threshold, best_far, best_frr, lowest_total_error

csv_sp = r'/home/gimicaroni/Documents/Unicamp/IC/HorusProjeto/Horus_part2/SamePerson_train.csv'
csv_dp = r'/home/gimicaroni/Documents/Unicamp/IC/HorusProjeto/Horus_part2/DifferentPerson_train.csv'

df_sp = pd.read_csv(csv_sp)
df_dp = pd.read_csv(csv_dp)

df_sp['Label'] = 1
df_dp['Label'] = 0

df = pd.concat([df_sp, df_dp])

unique_clusters = sorted(df['Cluster'].unique())

possible_thresholds = np.linspace(0.7, 1.2, 500)

thresh_per_cluster = {}
total_error_per_cluster = {}

for cluster in unique_clusters:
    ideal_threshold, far, frr, total_error = find_best_threshold_with_far(df, cluster, possible_thresholds, target_far=0.001)
    thresh_per_cluster[cluster] = ideal_threshold
    print(f"Cluster {cluster}: Number_of_images = {len(df[df['Cluster'] == cluster])} Ideal threshold = {ideal_threshold:.4f}, FAR = {far:.4f}, FRR = {frr:.4f}, Total error = {total_error}")
