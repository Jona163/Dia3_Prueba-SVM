import numpy as np

class SVM:
    """
    Implementación de un clasificador SVM (Support Vector Machine).
    """

    def __init__(self, tasa_aprendizaje=0.001, parametro_lambda=0.01, iteraciones=1000):
        """
        Inicializa el modelo SVM con los parámetros dados.
        
        Parámetros:
        - tasa_aprendizaje: tasa de aprendizaje para el descenso de gradiente.
        - parametro_lambda: parámetro de regularización.
        - iteraciones: número de iteraciones para el entrenamiento.
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.parametro_lambda = parametro_lambda
        self.iteraciones = iteraciones
        self.w = None  # Pesos
        self.b = None  # Sesgo

    def ajustar(self, X, y):
        """
        Ajusta el modelo SVM a los datos de entrada X y las etiquetas y.

        Parámetros:
        - X: matriz de características de tamaño (n_muestras, n_características).
        - y: vector de etiquetas de tamaño (n_muestras).
        """
        n_muestras, n_caracteristicas = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Reasigna etiquetas a -1 y 1

        # Inicialización de pesos y sesgo
        self.w = np.zeros(n_caracteristicas)
        self.b = 0

        # Descenso de gradiente para ajustar los pesos
        for _ in range(self.iteraciones):
            for idx, x_i in enumerate(X):
                condicion = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condicion:
                    # Actualización en caso de que el margen sea mayor o igual a 1
                    self.w -= self.tasa_aprendizaje * (2 * self.parametro_lambda * self.w)
                else:
                    # Actualización en caso contrario
                    self.w -= self.tasa_aprendizaje * (2 * self.parametro_lambda * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.tasa_aprendizaje * y_[idx]

    def predecir(self, X):
        """
        Predice las etiquetas para las muestras de entrada X.

        Parámetros:
        - X: matriz de características de tamaño (n_muestras, n_características).

        Retorna:
        - Predicciones: vector de etiquetas predichas.
        """
        aproximacion = np.dot(X, self.w) - self.b
        return np.sign(aproximacion)
