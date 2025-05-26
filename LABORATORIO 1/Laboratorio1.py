import os
import pandas as pd
# Computacion vectorial y cientifica para python
import numpy as np
# Librerias para graficación (trazado de gráficos)
from matplotlib import pyplot
#import matplotlib.pyplot as plt  #aumentado
from mpl_toolkits.mplot3d import Axes3D  # Necesario para graficar superficies 3D

# Verificar si el archivo existe
ruta_archivo = 'California_Houses.csv'  # Asegúrate de que el nombre sea exacto
if not os.path.exists(ruta_archivo):
    print(f"El archivo no existe en la ruta: {os.path.abspath(ruta_archivo)}")
else:
    print(f"El archivo existe. Cargando datos...")
    data = pd.read_csv(ruta_archivo)
    # Imprimir los nombres de las columnas para verificar
    print("Columnas disponibles en el dataset:")
    print(data.columns)
    # Seleccionar características (features) y variable objetivo (target)
    X = data[['Median_Income', 'Median_Age', 'Tot_Rooms', 'Tot_Bedrooms', 
                     'Population', 'Households', 'Latitude', 'Longitude', 
                     'Distance_to_coast', 'Distance_to_LA', 'Distance_to_SanDiego']]  # Columnas seleccionadas
    Y = data['Median_House_Value']  # Variable objetivo
    m = Y.size  # número de ejempores (filas)
    def  featureNormalize(X):

        X_norm = X.copy()
        mu = np.zeros(X.shape[1])
        sigma = np.zeros(X.shape[1])
        mu = np.mean(X, axis = 0)
        sigma = np.std(X, axis = 0)
        X_norm = (X - mu) / sigma
        return X_norm, mu, sigma
    # llama featureNormalize con los datos cargados
    X_norm, mu, sigma = featureNormalize(X)

    print(X)
    print('Media calculada:', mu)
    print('Desviación estandar calculada:', sigma)
    print(X_norm)
    # Añade el termino de interseccion a X
    # (Columna de unos para X0)
    X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
    def computeCostMulti(X, y, theta):

        # Inicializa algunos valores utiles
        m = y.shape[0] # numero de ejemplos de entrenamiento
        J = 0
        # h = np.dot(X, theta)
        J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
        return J
    def gradientDescentMulti(X, y, theta, alpha, num_iters):
        # Inicializa algunos valores
        m = y.shape[0] # numero de ejemplos de entrenamiento
        # realiza una copia de theta, el cual será acutalizada por el descenso por el gradiente
        theta = theta.copy()
        J_history = []
        for i in range(num_iters):
            theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
            J_history.append(computeCostMulti(X, y, theta))
        return theta, J_history
    # Elegir algun valor para alpha (probar varias alternativas)
    alpha = 0.001
    num_iters = 10000

    # inicializa theta y ejecuta el descenso por el gradiente
    theta = np.zeros(X.shape[1])
    theta, J_history = gradientDescentMulti(X, Y, theta, alpha, num_iters)

    # Grafica la convergencia del costo
    pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
    pyplot.xlabel('Numero de iteraciones')
    pyplot.ylabel('Costo J')

    # Muestra los resultados del descenso por el gradiente
    print('theta calculado por el descenso por el gradiente: {:s}'.format(str(theta)))

    # Estimar el precio para una casa de 1650 sq-ft, con 3 dormitorios
    X_array = [1, 2.34476576, 0.98214266, -0.804819, -0.970706, -0.974429, -0.977033, 1.052548, -1.327835, -0.635876, 1.158969, 1.165668]
    price = np.dot(X_array, theta)

    print('El precio estimado de este ejemplo (usando el descenso por el gradiente) es: ${:.0f}'.format(price))