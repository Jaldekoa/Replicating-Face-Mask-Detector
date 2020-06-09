import cv2
import tensorflow as tf

FONT = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

model = tf.keras.models.load_model('Detector Model.model')

while True:
    isTrue, img = cam.read()
    faces = face_cascade.detectMultiScale(img, 1.5, 3, minSize=(56,56))

    for (x, y, w, h) in faces:
        cara = img[y:y+h, x:x+w, :]
        cara = cv2.resize(img, (224, 224))
        cara = tf.keras.preprocessing.image.img_to_array(cara)
        cara = cara / 255.
        cara = tf.expand_dims(cara, axis=0)

        (con_masc, sin_masc) = model.predict(cara)[0]

        label = "Con mascarilla" if con_masc > sin_masc + 5 else "Sin mascarilla"
        color = (0, 255, 0) if label == "Con mascarilla" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(con_masc, sin_masc) * 100)

        cv2.rectangle(img, pt1 = (x, y), pt2 = (x + w, y + h), color = color, thickness = 2)
        cv2.putText(img, label, (x, y - 10), FONT, 0.5, color, 2)

    cv2.imshow("Replicating Face Mask Detector", img)

    k = cv2.waitKey(1)

    if k == 27: #Esc key
        break

cam.release()
cv2.destroyAllWindows()
