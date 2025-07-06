import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# thresholds específicos por cluster
# se quiser usar o fisico só substituir por exemplo: threshold = 0.7 
thresholds = {
    0: 0.8191,
    1: 0.6784,
    2: 0.7286,
    3: 0.8241,
    4: 0.7688,
    5: 0.6784,
    6: 0.6131,
    7: 0.7889,
    8: 0.7186,
    9: 0.6583,
    10: 0.5930,
    11: 0.7136,
    12: 0.6231,
    13: 0.6683,
    14: 0.7387
}

def verificar_acerto(distance, label, threshold):
    return int((distance <= threshold) == label)

csv_sp = r"C:\Users\emily\OneDrive\Documents\IC\parte 3 - testes\griaule\umd\SamePerson_train_umd.csv"
csv_dp = r"C:\Users\emily\OneDrive\Documents\IC\parte 3 - testes\griaule\umd\DifferentPerson_train_umd.csv"

df_sp = pd.read_csv(csv_sp)
df_dp = pd.read_csv(csv_dp)

df_sp['Label'] = 1
df_dp['Label'] = 0

df = pd.concat([df_sp, df_dp], ignore_index=True)

# matriz de confusão
VP_total, FP_total, VN_total, FN_total = 0, 0, 0, 0
acertos = []

for _, row in df.iterrows():
    distance = row['Cosine Distance']
    label = row['Label']
    cluster_a = row['Cluster A']
    cluster_b = row['Cluster B']

    t1 = thresholds.get(cluster_a, 0.6)
    t2 = thresholds.get(cluster_b, 0.6)
    threshold = max(t1, t2) # para usar o mínimo, substituir por min

    predicao = int(distance <= threshold)

    # matriz de confusão
    if label == 1 and predicao == 1:
        VP_total += 1
    elif label == 0 and predicao == 1:
        FP_total += 1
    elif label == 0 and predicao == 0:
        VN_total += 1
    elif label == 1 and predicao == 0:
        FN_total += 1

    acertos.append(verificar_acerto(distance, label, threshold))

taxa_acerto_geral = np.mean(acertos)
print(f"\nTaxa de Acerto Geral: {taxa_acerto_geral:.4f}")

print("\nMatriz de Confusão Geral:")
print("          Predito")
print("Real      Positivo  Negativo")
print(f"Positivo    {VP_total:4}      {FN_total:4}")
print(f"Negativo    {FP_total:4}      {VN_total:4}")

print("\nTaxa de Acerto por Cluster:")

# unir todos os clusters únicos existentes nas duas colunas
clusters_unicos = pd.unique(pd.concat([df['Cluster A'], df['Cluster B']]))

resultados = []

for cluster in sorted(clusters_unicos):
    df_cluster = df[(df['Cluster A'] == cluster) | (df['Cluster B'] == cluster)]
    acertos_cluster = []

    for _, row in df_cluster.iterrows():
        distance = row['Cosine Distance']
        label = row['Label']
        cluster_a = row['Cluster A']
        cluster_b = row['Cluster B']

        t1 = thresholds.get(cluster_a, 0.6)
        t2 = thresholds.get(cluster_b, 0.6)
        threshold = max(t1, t2) # para usar o mínimo, substituir por min

        acertos_cluster.append(verificar_acerto(distance, label, threshold))

    taxa = np.mean(acertos_cluster)
    resultados.append((cluster, taxa))

# desvio padrão
taxas = [taxa for _, taxa in resultados]
desvio_padrao = np.std(taxas)

# com distância à média
for cluster, taxa in resultados:
    distancia = taxa - media_geral
    print(f"Cluster {cluster:2}: {taxa:.4f}  (Distância da média: {distancia:+.4f})")

print(f"Desvio Padrão das Taxas: {desvio_padrao:.4f}")
