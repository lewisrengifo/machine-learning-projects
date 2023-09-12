# Importar las librerías necesarias
from keras.models import Sequential
from keras.layers import Dense, Activation
import argparse
import numpy as np
from pymongo import MongoClient

# Crear un analizador de argumentos
parser = argparse.ArgumentParser()

# Añadir un argumento para el rango de horas a predecir
parser.add_argument("horas", nargs="+", type=str, help="Rango de horas a predecir")

# Parsear los argumentos
args = parser.parse_args()

# Conectar a la base de datos
client = MongoClient("3.89.102.243", 27017)
db = client.temperaturas
collection = db.tempSensed

# Crear una lista para almacenar los valores de x_train e y_train
x_train = []
y_train = []
# Recorrer la lbase de datos
for document in collection.find():
    # Obtener la hora, los minutos y los segundos
    hora = document["hour"]
    minutos = document["minute"]
    segundos = document["second"]
    temperaturas = document["temperatureSensed"]
    x = np.array([hora, minutos, segundos], dtype="float32")
    y = np.array([temperaturas], dtype="float32")

    # Agregar el array a x_train
    x_train.append(x)
    y_train.append(y)

# Convertir la lista a un array
x_train = np.array(x_train)

# Crear una lista para almacenar los valores de y_train
y_train = np.array(y_train)

# Inicializar la red neuronal
model = Sequential()

# Capa de entrada
model.add(Dense(units=64, input_dim=3))

# Capa oculta
model.add(Dense(units=64))
model.add(Activation("relu"))

# Capa de salida
model.add(Dense(units=1))

# Compilar el modelo
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

# Entrenar la red neuronal con los datos de temperatura previamente medidos
model.fit(x_train, y_train, epochs=100, batch_size=128)

# Crear una lista para almacenar los valores de x_test
x_test = []

# Recorrer la lista de horas, minutos y segundos a predecir
# Recorrer el rango de horas a predecir
hora = int(hora)
for hora in args.horas:
    # Agregar la hora a x_test
    x = np.array([hora, 0, 0], dtype="float32")

    # Agregar el array a x_test
    x_test.append(x)

# Convertir la lista a un array
x_test = np.array(x_test)

# Realizar predicciones utilizando el modelo entrenado
predictions = model.predict(x_test)
print(x_test)
print("las predicciones de la temperatura son las siguientes: ")
print(predictions)

# Guardar las predicciones en la base de datos
# Crear una colección llamada "predictions" en la base de datos
predictions_collection = db.predictions

predictions_list = []
for prediction, time in zip(predictions, x_test):
    prediction_dict = {
        "prediction": prediction.item(0),
        "time": time.item(0)
    }
    predictions_list.append(prediction_dict)

print(predictions_list)
# Crear un documento que contenga la variable predictions
for prediction in predictions_list:
    predictions_document = {
        "predictions": predictions_list
    }

# Insertar el documento en la colecciónn
predictions_collection.insert_one(predictions_document)

client.close()
