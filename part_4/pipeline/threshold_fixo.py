import pandas as pd
import numpy as np

def verificar_acerto(distance, label, threshold):
    return int((distance <= threshold) == label)

csv_sp = r".csv"
csv_dp = r".csv"

threshold_escolhido = 0.7

# Carregar dados
df_sp = pd.read_csv(csv_sp)
df_dp = pd.read_csv(csv_dp)

df_sp['Label'] = 1
df_dp['Label'] = 0

df = pd.concat([df_sp, df_dp], ignore_index=True)

# threshold fixo
VP_total, FP_total, VN_total, FN_total = 0, 0, 0, 0
acertos = []

# análise por cluster individual
acertos_por_cluster = {}

for idx, row in df.iterrows():
    distance = row['Cosine Distance']
    label = row['Label']
    pred = int(distance <= threshold_escolhido)
    acerto = verificar_acerto(distance, label, threshold_escolhido)
    acertos.append(acerto)

    # matriz de confusão
    if label == 1 and pred == 1:
        VP_total += 1
    elif label == 0 and pred == 1:
        FP_total += 1
    elif label == 0 and pred == 0:
        VN_total += 1
    elif label == 1 and pred == 0:
        FN_total += 1

    # contagem por cluster (tanto A quanto B)
    for cluster_col in ['Cluster A', 'Cluster B']:
        cluster = row[cluster_col]
        if cluster not in acertos_por_cluster:
            acertos_por_cluster[cluster] = []
        acertos_por_cluster[cluster].append(acerto)

taxa_geral = np.mean(acertos)
desvio_padrao = np.std(acertos)

print(f"\nAvaliação com Threshold fixo = {threshold_escolhido:.4f}")
print(f"Taxa de Acerto Geral: {taxa_geral:.4f}")
print(f"Desvio Padrão da Taxa de Acerto: {desvio_padrao:.4f}")

# comparação por cluster individual
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
