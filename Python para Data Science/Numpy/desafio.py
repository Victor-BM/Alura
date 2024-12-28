import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def regressao_linear(X, Y):
    '''Calcula a reta de regressão linear via MMQ'''
    n = np.size(Y)
    a = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y))/(n*np.sum(X**2) - np.sum(X)**2)
    b = np.mean(Y) - a*np.mean(X)
    y = a*X + b
    return y

def plotar_laranja(laranja_diametro, laranja_peso, y_laranja):
    '''Plota o gráfico da Laranja real e normalizado'''
    plt.plot(laranja_diametro, laranja_peso, color = 'orange', label = 'Laranja real')
    plt.plot(laranja_diametro, y_laranja, color = 'green', linestyle = 'dashed', label = 'Laranja normalizada')
    plt.legend()
    plt.xlabel('Diâmetro')
    plt.ylabel('Peso')
    plt.text(0.99, 0.05, f'Normalização: {np.linalg.norm(laranja_peso-y_laranja):.2f}', ha='right', va='top', transform=plt.gca().transAxes)
    plt.show()

def plotar_toranja(toranja_diametro, toranja_peso, y_toranja):
    '''Plota o gráfico da Toranja real e normalizado'''
    plt.plot(toranja_diametro, toranja_peso, color = 'red', label = 'Toranja normalizada')
    plt.plot(toranja_diametro, y_toranja, color = 'blue', linestyle = 'dashed', label = 'Toranja normalizado')
    plt.legend()
    plt.xlabel('Diâmetro')
    plt.ylabel('Peso')
    plt.text(0.99, 0.05, f'Normalização: {np.linalg.norm(toranja_peso-y_toranja):.2f}', ha='right', va='top', transform=plt.gca().transAxes)
    plt.show()

def plotar_comparado(laranja_diametro, laranja_peso, toranja_diametro, toranja_peso):
    '''Plota o gráfico comparando Toranjas e Laranjas reais'''
    plt.plot(laranja_diametro, laranja_peso, color = 'orange', label = 'Laranja')
    plt.plot(toranja_diametro, toranja_peso, color = 'red', label = 'Toranja')
    plt.legend()
    plt.xlabel('Diâmetro')
    plt.ylabel('Peso')
    plt.show()

def coef_ang_aleatorio(X, Y, nome):
    '''Calcula o coeficiente angular com numeros aleatorios'''
    b = 17
    np.random.seed(2931)
    coef_angulares = np.random.uniform(low = 0, high = 30, size = 100)
    norma = np.array([])
    for i in range(100):
        norma = np.append(norma, np.linalg.norm(Y-(coef_angulares[i]*X+b)))
    print(coef_angulares[np.argmin(norma)], norma[np.argmin(norma)])
    dados = np.column_stack([norma ,coef_angulares])
    np.savetxt(f'dados_{nome}.csv', dados , delimiter=',', header='Norma,Coef_Ang')

def main():
    url = 'https://raw.githubusercontent.com/allanspadini/numpy/dados/citrus.csv'
    arquivo = pd.read_csv(url)
    dado = np.loadtxt(url, delimiter=',', usecols=np.arange(1, len(arquivo.columns), 1), skiprows=1)

    #definindo dados
    laranja_diametro = dado[:5000, 0]
    laranja_peso = dado[:5000, 1]
    toranja_diametro = dado[5000:, 0]
    toranja_peso  =dado[5000:, 1]

    #ajuste linear
    y_laranja = regressao_linear(laranja_diametro, laranja_peso)
    y_toranja = regressao_linear(toranja_diametro, toranja_peso)
    
    #ajuste linear aleatorio
    coef_ang_aleatorio(laranja_diametro, laranja_peso, 'laranja')
    coef_ang_aleatorio(toranja_diametro, toranja_peso, 'toranja')

    #plotando os gráficos
    plotar_laranja(laranja_diametro, laranja_peso, y_laranja)
    plotar_toranja(toranja_diametro, toranja_peso, y_toranja)
    plotar_comparado(laranja_diametro, laranja_peso, toranja_diametro, toranja_peso)

if __name__ == '__main__':
    main()
