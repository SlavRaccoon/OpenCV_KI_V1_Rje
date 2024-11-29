import cv2
import time
import keyboard
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(1)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:

    result, video_frame = video_capture.read() 
    if result is False:
        break 

    faces = detect_bounding_box(video_frame)

    if cv2.waitKey(8) & 0xFF == ord("r"):
        for x in range(0,480):
            for y in range(0, 640):
                c = video_frame[x,y][2]
                video_frame[x,y] = [0, 0, c]
    if cv2.waitKey(8) & 0xFF == ord("g"):
        if keyboard.is_pressed("g"):
            for x in range(0,480):
                for y in range(0, 640):
                    c = video_frame[x,y][1]
                    video_frame[x,y] = [0, c, 0]
    if cv2.waitKey(8) & 0xFF == ord("b"):
        if keyboard.is_pressed("b"):
            for x in range(0,480):
                for y in range(0, 640):
                    c = video_frame[x,y][0]
                    video_frame[x,y] = [c, 0, 0]
    

    cv2.imshow("My Face Detection Project", video_frame) 
    #print(time.time())
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()