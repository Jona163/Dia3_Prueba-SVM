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
