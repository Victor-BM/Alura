import numpy as np
import matplotlib.pyplot as plt

# Gerar uma sequência de valores de x de -1 a 1
x = np.arange(-1, 1.01, 0.0001)

# Implementação da fórmula
y1 = np.sqrt(1 - x**2)
y2 = -np.sqrt(1 - x**2)

# Plotar o gráfico com as duas partes do círculo
plt.plot(x, y1, 'r')
plt.plot(x, y2, 'r' )

# Adicionar o título do gráfico e os rótulos dos eixos x e y
plt.title("Círculo")
plt.xlabel("Eixo x")
plt.ylabel("Eixo y")

# Exibir o gráfico
plt.show()