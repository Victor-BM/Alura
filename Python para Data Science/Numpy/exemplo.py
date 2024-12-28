import numpy as np
import matplotlib.pyplot as plt
url = 'https://raw.githubusercontent.com/alura-cursos/numpy/dados/apples_ts.csv'
dado = np.loadtxt(url,delimiter=',',usecols=np.arange(1,88,1))
dado_transporsto = dado.T
datas = dado_transporsto[:,0]
tempo = np.arange(1, 88, 1)
preços = dado_transporsto[:,1:6]
Moscow = preços[:, 0]
Kaliningrad = preços[:, 1]
Moscow_ano1 = Moscow[0:12] 
Moscow_ano2 = Moscow[12:24]
Moscow_ano3 = Moscow[24:36]
Moscow_ano4 = Moscow[36:48]
plt.plot(np.arange(1, 13, 1), Moscow_ano1)
plt.plot(np.arange(1, 13, 1), Moscow_ano2)
plt.plot(np.arange(1, 13, 1), Moscow_ano3)
plt.plot(np.arange(1, 13, 1), Moscow_ano4)
plt.legend(['ano1', 'ano2', 'ano3', 'ano4'])
plt.show()
print(np.array_equal(Moscow_ano3, Moscow_ano4))
print(np.allclose(Moscow_ano3, Moscow_ano4, 10))
print(sum(np.isnan(Kaliningrad)))
plt.plot(tempo, Kaliningrad)
plt.show()
Kaliningrad[4] = np.mean([Kaliningrad[3], Kaliningrad[5]])
x = tempo
y = 0.52*x + 80
np.linalg.norm(Moscow-y)
Y = Moscow
X = tempo
n = np.size(Moscow)
#calculando o coefieciente angular
a = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y))/(n*np.sum(X**2) - np.sum(X)**2)
#calculando o coeficiente linear
b = np.mean(Y) - a*np.mean(X)
y_dois = a*X + b
print(np.linalg.norm(Moscow-y_dois))
plt.plot(tempo, Moscow)
plt.plot(X, y_dois)
plt.plot(41.5,41.5*a+b,'*r')
plt.plot(100,100*a+b,'*r')
plt.plot(X, y)
plt.show()
np.random.seed(84)
coef_angulares = np.random.uniform(low=0.10,high=0.90,size=100)
norma2 = np.array([])
for i in range(100):
    norma2 = np.append(norma2, np.linalg.norm(Moscow-(coef_angulares[i]*X+b)))
dados = np.column_stack([norma2,coef_angulares])
np.savetxt('dados.csv',dados,delimiter=',')
