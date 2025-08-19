import pandas as pd
import numpy as np

# contar FP e FN dado um threshold
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

# encontrar melhor threshold por cluster
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

def verificar_acerto(distance, label, threshold):
    return int((distance <= threshold) == label)

csv_sp = r".csv"
csv_dp = r".csv"

df_sp = pd.read_csv(csv_sp)
df_dp = pd.read_csv(csv_dp)

df_sp['Label'] = 1
df_dp['Label'] = 0

# converter Cluster A e B para numérico
df_sp['Cluster A'] = pd.to_numeric(df_sp['Cluster A'], errors='coerce')
df_sp['Cluster B'] = pd.to_numeric(df_sp['Cluster B'], errors='coerce')
df_dp['Cluster A'] = pd.to_numeric(df_dp['Cluster A'], errors='coerce')
df_dp['Cluster B'] = pd.to_numeric(df_dp['Cluster B'], errors='coerce')

# cria coluna 'Cluster' como média aritmética
df_sp['Cluster'] = ((df_sp['Cluster A'] + df_sp['Cluster B']) / 2).round(0).astype(int)
df_dp['Cluster'] = ((df_dp['Cluster A'] + df_dp['Cluster B']) / 2).round(0).astype(int)

df = pd.concat([df_sp, df_dp], ignore_index=True)

# calcular thresholds ideais por cluster
unique_clusters = sorted(df['Cluster'].unique())
possible_thresholds = np.linspace(0, 1, 200)

thresholds = {}
print("Threshold ótimo por cluster:")
for cluster in unique_clusters:
    ideal_threshold, total_error = find_best_threshold(df, cluster, possible_thresholds)
    thresholds[cluster] = ideal_threshold
    print(f"Cluster {cluster}: Ideal threshold = {ideal_threshold:.4f}, Total error (FP + FN) = {total_error}")

VP_total, FP_total, VN_total, FN_total = 0, 0, 0, 0
acertos = []

for _, row in df.iterrows():
    distance = row['Cosine Distance']
    label = row['Label']
    cluster = row['Cluster']
    threshold = thresholds.get(cluster, 0.6)

    pred = int(distance <= threshold)

    if label == 1 and pred == 1:
        VP_total += 1
    elif label == 0 and pred == 1:
        FP_total += 1
    elif label == 0 and pred == 0:
        VN_total += 1
    elif label == 1 and pred == 0:
        FN_total += 1

    acertos.append(verificar_acerto(distance, label, threshold))

taxa_geral = np.mean(acertos)
desvio_padrao = np.std(acertos)

print(f"\nTaxa de Acerto Geral: {taxa_geral:.4f}")
print(f"Desvio Padrão da Taxa de Acerto: {desvio_padrao:.4f}")

print("\nTaxa de Acerto por Cluster (A ou B):")
for cluster_id in sorted(acertos_por_cluster):
    lista = acertos_por_cluster[cluster_id]
    taxa_cluster = np.mean(lista)
    distancia = taxa_cluster - taxa_geral
    sinal = '+' if distancia >= 0 else ''
    print(f"Cluster {cluster_id:2}: {taxa_cluster:.4f}  | {sinal}{distancia:.4f}")

print("\nMatriz de Confusão:")
print("          Predito")
print("Real      Positivo  Negativo")
print(f"Positivo    {VP_total:4}      {FN_total:4}")
print(f"Negativo    {FP_total:4}      {VN_total:4}")
