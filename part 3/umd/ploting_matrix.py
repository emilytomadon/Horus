import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(matriz_confusao):
    
    porcentagens = [f'{value:.2%}' for value in\
                        matriz_confusao.flatten()/np.sum(matriz_confusao)]
    contagem = [f'{value:0.0f}' for value in matriz_confusao.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(contagem, porcentagens)]
    labels = np.asarray(labels).reshape(2,2)
    print(labels)
    sns.heatmap(matriz_confusao, annot=labels, fmt='', cmap='Blues')
    plt.title('Matriz de confusão - UMD Mínima Soma')
    plt.xlabel('Previsão')
    plt.ylabel('Valor real')
    plt.show()

if __name__ == "__main__":
    matriz_confusao = np.array([[9557,443],
                                [1,9999]])
    plot_heatmap(matriz_confusao)