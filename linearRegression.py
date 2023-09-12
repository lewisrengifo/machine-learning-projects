import numpy as np

# Inicializamos los pesos del modelo con valores aleatorios
w = np.random.randn(2)

# Definimos la función de pérdida, que en este caso es la suma de los errores cuadráticos
def loss(x, y, w):
  y_pred = x * w[0] + w[1]
  error = y - y_pred
  return np.sum(error**2)

# Definimos la función de entrenamiento, que actualizará los pesos del modelo
# en base a los datos de entrada y las etiquetas
def train(x, y, w, learning_rate):
  y_pred = x * w[0] + w[1]
  error = y - y_pred
  w[0] += learning_rate * np.sum(error * x)
  w[1] += learning_rate * np.sum(error)
  return w

# Cargamos los datos de entrenamiento
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 3, 4, 5, 6])

# Entrenamos el modelo utilizando una tasa de aprendizaje de 0.01
w = train(x_train, y_train, w, 0.01)

# Hacemos predicciones utilizando el modelo entrenado
x_test = np.array([6, 7, 8, 9, 10])
y_pred = x_test * w[0] + w[1]

# Calculamos el error en las predicciones
#error = y_test - y_pred

# Imprimimos los resultados
print("w:", w)
print("y_pred:", y_pred)
#print("error:", error)
