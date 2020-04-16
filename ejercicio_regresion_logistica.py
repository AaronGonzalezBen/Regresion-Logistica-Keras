# regresión logistica que permita determinar qué estudiantes van a pasar
# el examen según sus horas de estudio y de sueno

# Set de datos: estudiantes que aprobaron (1) o reprobaron (0)
# el examen con base en el número de horas estudiadas (x1) y el # de 
# horas de sueno (x2)

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Leer el dataset y dibujar la nube de puntos con categorías

datos = pd.read_csv('dataset.csv', sep = ",")
print(datos)

# Crear datos de entrentamiento (X) y categorías de salida (Y)
# X: las dos primeras columnas
# Y: la tercera columna

X = datos.values[:,0:2]
Y = datos.values[:,2]

# Graficar las categorías
idx0 = np.where(Y==0)
idx1 = np.where(Y==1)

plt.scatter(X[idx0,0], X[idx0,1], color = 'red', label = 'Categoría = 0')
plt.scatter(X[idx1,0], X[idx1,1], color = 'gray', label = 'Categoría = 1')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(bbox_to_anchor = (0.765,0.6), fontsize = 8, edgecolor = 'black')
plt.title('Datos originales')
plt.show()

# Implementación del modelo de regresión logistica a partir de una neurona
# artificial

"""  
- input_dim: 2 - segun las caracteristicas de los datos (x1, x2)
- output_dim: 1 - se tiene una de dos posibles categorías (0 o 1)
- Activación: sigmoidal (oscila entre 0 y 1)
"""

np.random.seed(2)
input_dim = X.shape[1] # numero de columnas de X (2)
output_dim = 1

modelo = Sequential()
modelo.add(Dense(output_dim, input_dim = input_dim, activation = 'sigmoid'))

# Optimización: se usará el gradiente descendente (SGD) con un lr = 0.2
# Función de error: entropía cruzada (binary_crossentropy)
# Métrica de desempeno: accuracy (precisión) - de todos los datos se
# calcula la relación de cuantos fueron clasificados correctamente

""" 
Nota: La entropía cruzada se aplica mejor en problemas de clasificación, 
el ECM aplica mejor para problemas de regresión
"""

sgd = SGD(lr = 0.2)
modelo.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Entrenamiento

num_epochs = 1000
batch_size = X.shape[0]     # Todo el conjunto de datos
historia = modelo.fit(X,Y, epochs = num_epochs, batch_size = batch_size, verbose = 2)

# Graficar el comportamiento de la pérdida y de la precisión
plt.subplot(1,2,1)
plt.plot(historia.history['loss'])
plt.ylabel('Pérdida')
plt.xlabel('Epoch')
plt.title('Comportamiento de la pérdida')

plt.subplot(1,2,2)
plt.plot(historia.history['accuracy'])
plt.ylabel('Precisión')
plt.xlabel('Epoch')
plt.title('Comportamiento de la precisión')

ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.show()

# Graficamos la frontera de decisión

def dibujar_frontera(X,Y,modelo,titulo):
    # Valor minimo y maximo rellenado con ceros
    x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
    y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
    h = 0.01

    # Grilla de puntos
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predecir categorias para cada punto en la grilla
    Z = modelo.predict_classes(np.c_[xx.ravel(), yy.ravel()])   # predict_classes predice directamente el clasificador según los puntos, predict entrega la probabilidad de las clasificaciones
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap = plt.cm.Set1, alpha = 0.8)

    idx0 = np.where(Y==0)
    idx1 = np.where(Y==1)
    plt.scatter(X[idx0,0],X[idx0,1], color = 'red', edgecolor = 'k', label = 'Categoría: 0')
    plt.scatter(X[idx1,0],X[idx1,1], color = 'gray', edgecolor = 'k', label = 'Categoría: 1')
    plt.legend(bbox_to_anchor = (0.765,0.6), fontsize = 8, edgecolor = 'black')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(titulo)

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()

dibujar_frontera(X,Y,modelo, 'Frontera de decisión después del entrenamiento')