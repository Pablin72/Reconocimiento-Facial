import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Cargar el modelo entrenado
modelo_reconocimiento_facial = load_model("modeloCNNv6.h5")

# Supongamos que tienes una lista de nombres de personas
lista_nombres_personas = ['Sergio', 'Sebastian', 'Anthony', 'Mattew', 'Diego', 'Fernando', 'Desconocidos', 'Pablo']

def identify_person(image_path, model):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make predictions with the model
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)

    return predicted_label

@app.route("/predict", methods=["POST"])
def predict_person():
    # Obtiene la ruta de la imagen de la solicitud
    imagen_path = request.json.get("imagen_path")

    # Realizar la predicción
    persona_predicha = identify_person(imagen_path, modelo_reconocimiento_facial)

    # Obtener el nombre de la persona predicha
    nombre_persona_predicha = lista_nombres_personas[persona_predicha]

    # Devolver la predicción como respuesta JSON
    print("Persona predicha:", nombre_persona_predicha)
    return jsonify({"persona_predicha": nombre_persona_predicha})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
