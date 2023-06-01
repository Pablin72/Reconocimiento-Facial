import cv2
import os
import imutils
from PIL import Image

personName = 'Fernando'
dataPath = 'C:/Users/pdarc/Desktop/Reconocimiento Facial Recortando Rostros/Fotos'  # Cambia a la ruta donde hayas almacenado Data
personPath = os.path.join(dataPath, personName)

if not os.path.exists(personPath):
    print('Carpeta creada:', personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture('Video.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 500

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Guardar rostro utilizando PIL
        image_path = os.path.join(personPath, 'rostro_{}.jpg'.format(count))
        pil_image = Image.fromarray(cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB))
        pil_image.save(image_path)

        print('Guardado:', image_path)
        count += 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 1000:
        break

cap.release()
cv2.destroyAllWindows()
