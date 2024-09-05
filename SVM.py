# Autor: Jonathan Hernández
# Fecha: 04 Septiembre 2024
# Descripción: Código para una simulación de batalla Pokémon de estilo retro
# GitHub: https://github.com/Jona163

#Importacion libreria de Numpy
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


# Pruebas
if __name__ == "__main__":
    # Importaciones necesarias
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # Creación de datos de prueba
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)  # Reasignación de etiquetas

    # División de los datos en conjunto de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Creación y ajuste del modelo SVM
    clasificador = SVM()
    clasificador.ajustar(X_entrenamiento, y_entrenamiento)

    # Predicciones con el conjunto de prueba
    predicciones = clasificador.predecir(X_prueba)

    # Función para calcular la precisión
    def precision(y_real, y_pred):
        precision = np.sum(y_real == y_pred) / len(y_real)
        return precision

    # Impresión de la precisión del modelo
    print("Precisión del clasificador SVM:", precision(y_prueba, predicciones))

    # Visualización del SVM y los hiperplanos
    def visualizar_svm():
        """
        Visualiza los datos, el hiperplano y las márgenes del clasificador SVM.
        """
        def obtener_valor_hiperplano(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        # Hiperplano
        x1_1 = obtener_valor_hiperplano(x0_1, clasificador.w, clasificador.b, 0)
        x1_2 = obtener_valor_hiperplano(x0_2, clasificador.w, clasificador.b, 0)

        # Márgenes
        x1_1_m = obtener_valor_hiperplano(x0_1, clasificador.w, clasificador.b, -1)
        x1_2_m = obtener_valor_hiperplano(x0_2, clasificador.w, clasificador.b, -1)

        x1_1_p = obtener_valor_hiperplano(x0_1, clasificador.w, clasificador.b, 1)
        x1_2_p = obtener_valor_hiperplano(x0_2, clasificador.w, clasificador.b, 1)

        # Graficar hiperplano y márgenes
        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        # Límites del gráfico
        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    visualizar_svm()
