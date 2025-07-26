import cv2
import os

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def capture_face(name):
    cam = cv2.VideoCapture(0)
    count = 0
    path = 'student_images/' + name
    os.makedirs(path, exist_ok=True)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        faces = face_classifier.detectMultiScale(frame, 1.3, 5)
        for (x,y,w,h) in faces:
            count += 1
            face = cv2.resize(frame[y:y+h, x:x+w], (200, 200))
            cv2.imwrite(f"{path}/{count}.jpg", face)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow("Registering Face", frame)
        if cv2.waitKey(1) == 13 or count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Face capture complete for {name}")

if __name__ == "__main__":
    username = input("Enter student name: ")
    capture_face(username)
