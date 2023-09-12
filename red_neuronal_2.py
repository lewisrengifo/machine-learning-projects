# Importar las librerías necesarias
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation
import argparse
import numpy as np

from pymongo import MongoClient

# Conectar a la base de datos
client = MongoClient("localhost", 27017)
db = client.temperaturas
collection = db.tempSensed

# Crear una lista para almacenar los valores de x_train
x_train = []

# Crear una lista para almacenar los valores de y_train
y_train = []

# Obtener los documentos de la colección "mediciones"


# Cerrar la conexión a la base de datos


# Crear un analizador de argumentos
parser = argparse.ArgumentParser()

# Añadir un argumento para el rango de horas a predecir
parser.add_argument("horas", nargs="+", type=int, help="Rango de horas a predecir")

# Parsear los argumentos
args = parser.parse_args()

# Crear una lista para almacenar los valores de x_test
x_test = []

# Recorrer el rango de horas a predecir
for hora in args.horas:
    # Agregar la hora a x_test
    x_test.append(hora)

x_test = np.array(x_test)

# Inicializar la red neuronal
model = Sequential()

# Capa de entrada
model.add(Dense(units=64, input_dim=3))

# Capa oculta
model.add(Dense(units=64))
model.add(Activation('relu'))

# Capa de salida
model.add(Dense(units=3))

# Compilar el modelo
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Entrenar la red neuronal con los datos de temperatura previamente medidos
x_train = []
y_train = []
for document in collection.find():
    # Obtener la hora, los minutos y los segundos
    hora = document["hour"]
    minutos = document["minute"]
    segundos = document["second"]

    # Crear un array con la hora, los minutos y los segundos
    x = np.array([hora, minutos, segundos], dtype="float32")

    # Obtener la temperatura medida
    temperatura = document["temperatureSensed"]

    # Agregar el array a x_train
    x_train.append(x)
    y_train.append(temperatura)

client.close()
x_train = np.array(x_train)
print(x_train)
print(y_train)


model.fit(x_train, y_train, epochs=100, batch_size=128)

# Realizar predicciones utilizando el modelo entrenado
#x_test = [18, 19, 20, 21, 22]
predictions = model.predict(x_test)

print(predictions)
