import cv2
import tkinter as tk
from PIL import Image, ImageTk
import requests

api_url = "http://localhost:5000/predict"  # Cambia la URL de la API según donde esté alojada

def take_photo():
    global frame, photo_taken, predicted_label

    reset_photo()  # Restablecer la variable photo_taken a False para permitir tomar una nueva foto

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        rostro = frame[y:y+h, x:x+w]

        # Guardar rostro utilizando OpenCV
        image_path = "Predecir.jpg"
        cv2.imwrite(image_path, rostro)

        # Enviar la ruta de la foto al API para la predicción
        predicted_label = send_prediction_request(image_path)

        photo_taken = True

        btn_exit.config(state=tk.NORMAL)
    else:
        print("No se detectó ningún rostro en la foto.")

def reset_photo():
    global photo_taken
    photo_taken = False

def send_prediction_request(image_path):
    data = {"imagen_path": image_path}
    response = requests.post(api_url, json=data)

    if response.status_code == 200:
        data = response.json()
        nombre_persona_predicha = data.get("persona_predicha")
        print("Persona predicha:", nombre_persona_predicha)
        return nombre_persona_predicha
    else:
        print("Error en la solicitud al API.")
        return "Desconocido"

def exit_app():
    global cap
    cap.release()
    root.destroy()

def update_frame():
    global frame, predicted_label
    ret, frame = cap.read()
    if ret:
        frame_copy = frame.copy()  # Crear una copia del frame para dibujar el rectángulo sin modificar el frame original
        frame_copy = cv2.resize(frame_copy, (640, 480))
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if photo_taken:
                # Dibujar el rectángulo alrededor del rostro y mostrar el nombre de la persona predicha
                cv2.putText(frame_copy, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        lbl_camera.imgtk = photo
        lbl_camera.configure(image=photo)
    lbl_camera.after(10, update_frame)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

photo_taken = False
predicted_label = ""

root = tk.Tk()
root.title("Capturador de Fotos")
root.geometry("800x600")

lbl_camera = tk.Label(root)
lbl_camera.pack(pady=20)

btn_take_photo = tk.Button(root, text="Predecir", command=take_photo)
btn_take_photo.pack(pady=10)

btn_exit = tk.Button(root, text="Salir", command=exit_app, state=tk.DISABLED)
btn_exit.pack(pady=10)

# Iniciar la actualización del frame antes de entrar al mainloop
update_frame()

root.mainloop()
