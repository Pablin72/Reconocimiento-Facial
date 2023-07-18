from flask import Flask, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/reconocimiento-facial', methods=['POST'])
def reconocimiento_facial():
    IMG_W = 150
    IMG_H = 150

    # Cargar el modelo pre-entrenado
    model = load_model('modeloCNNv3.h5')

    # Cargar las etiquetas de las personas (reemplazar con tus propias etiquetas)
    personas = ['Sebastian', 'Sergio', 'Pablo', 'Anthony', 'Mattew', 'Diego', 'Fernando']

    # Cargar el clasificador Haar Cascade para la detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Función para preprocesar una imagen de rostro
    def preprocess_face(image):
        image = cv2.resize(image, (IMG_W, IMG_H))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    # Función para realizar el reconocimiento facial en tiempo real
    def recognize_faces():
        # Inicializar la cámara
        cap = cv2.VideoCapture(0)

        while True:
            # Leer el frame de la cámara
            ret, frame = cap.read()


            # Detectar rostros en el frame
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extraer el rostro de la región detectada
                face_img = frame[y:y+h, x:x+w]

                # Preprocesar la imagen del rostro
                preprocessed_face = preprocess_face(face_img)

                # Realizar la predicción del rostro
                predictions = model.predict(preprocessed_face)
                predicted_label_index = np.argmax(predictions)
                predicted_label = personas[predicted_label_index]

                # Dibujar el cuadro y la etiqueta en el frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Mostrar el frame resultante
            cv2.imshow('Reconocimiento Facial', frame)

            # Salir del bucle si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Liberar los recursos y cerrar las ventanas
        cap.release()
        cv2.destroyAllWindows()

    # Ejecutar el reconocimiento facial en tiempo real
    recognize_faces()


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0')
