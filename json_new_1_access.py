import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

import cv2

import vlc
import random
import glob


facecascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

print("lets get started.....")
cv2.namedWindow("Python screenshot app")

img_counter = random.randint(1, 1000000)

# saving the image with different name we are using random function to generate different integer values
# print(img_counter)

print("hit spacebar to capture the image")
print("hit Escape to close the app")


while True:
    result, frame = cam.read()
    # to remove the mirroring of camera images
    frame = cv2.flip(frame, 1)

    # start reading the camera
    faces = facecascade.detectMultiScale(frame, 1.1, 4)

    # function will detect the faces in the live video images
    # to give frames/rectagular frames for the multiple faces we are using for loop with its 4 co-ordinates

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 6, 100), 2)

    # if camear fail to capture the image below line will be printed
    if not result:
        print("failed to grab frames")
        break
    # it will make the camera-screen appear to the user
    cv2.imshow("test", frame)
    # we are mentaining a key value for the next process
    # press Ecs will end,space will capture the image
    k = cv2.waitKey(1)

    if k % 256 == 27:
        print("closing the app")
        exit()

    elif k % 256 == 32:
        # imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # img = cv2.imread(imgGray)

        img_name = "screenshot\opencv_capture-{}.png".format(img_counter)
        # we are saving colored image into one file called screenshot
        cv2.imwrite(img_name, frame)
        print("screenshot taken")
        cam.release()
        # proceeding with captured image we are croping the face in the image and saving into a folder called faces
        img = cv2.imread(img_name)

        faces = facecascade.detectMultiScale(img, 1.1, 5)
        # looping through multiple faces
        for face in faces:
            x, y, w, h = face
            # print(face)
            face_rectangle = img[y:y + h, x:x + w]
            # converting colored image to grayscale
            imgGray = cv2.cvtColor(face_rectangle, cv2.COLOR_BGR2GRAY)
            imgGray = cv2.resize(imgGray, (48, 48))

            #image_path = "faces/croped-{}.jpeg".format(random.randint(0, 100000))
            # saving at destination folder
            #cv2.imwrite(image_path, imgGray)
            # displaying gray-scale-face
            #cv2.imshow("image", imgGray)
            #print("Croped image saved to file")
            model = model_from_json(open("fer.json_new_4", "r").read())

            model.load_weights('fer_new_4.h5')
            #print(imgGray)
            #img = cv2.imread(imgGray)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # img = cv2.resize(img,(48,48))
            img_pixels = image.img_to_array(imgGray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            prediction = model.predict(img_pixels)

            max_index = np.argmax(prediction[0])
            print(max_index)

            emotion = ('happy','sad')#,'neutral')
            predicted_emotion=emotion[max_index]

            #print(predicted_emotion)

        break

cv2.waitKey(0)
cv2.destroyAllWindows()

#print("........to stop the song press 0.........")
print(predicted_emotion)
x= predicted_emotion

if x == 'happy':
    # glob used to access any file in a folder
    path = glob.glob("C:\\Users\\SEEMA\\PycharmProjects\\opencvpython\\music\\happy\\*.mp3")

    # to choose a random number we mentain a count
    count=random.randint(0,19)
    # c is for loop counter
    c=0
    for file in path:
        # when loop counter c is equal to randam value of count
        if c == count:
            # mediaplayer used to play audio of given path
            audio = vlc.MediaPlayer(file)
            audio.play()
            print("to stop the song press '0'...... ")

            k = int(input())
            # when we press 0 audio stops
            if k == 0:
                print("stoped playing")
                audio.stop()
                exit()
        c=c+1
# check and go in this block if user is sad
if x == 'sad':
    path = glob.glob("C:\\Users\\SEEMA\\PycharmProjects\\opencvpython\\music\\sad\\*.mp3")
    # to choose a random number we mentain a count
    count = random.randint(0, 16)
    c = 0
    for file in path:
        if c == count:
            audio = vlc.MediaPlayer(file)
            audio.play()
            print("to stop the song press '0'...... ")

            k = int(input())
            if k == 0:
                print("stoped playing")
                audio.stop()
                exit()
        c = c + 1


# check and go in this block if user is neutral
if x == 'neutral':
    path = glob.glob("C:\\Users\\SEEMA\\PycharmProjects\\opencvpython\\music\\neutral\\*.mp3")
    # to choose a random number we mentain a count
    count = random.randint(0, 19)
    c = 0
    for file in path:
        if c == count:
            audio = vlc.MediaPlayer(file)
            audio.play()
            print("to stop the song press '0'...... ")

            k = int(input())
            if k == 0:
                print("stoped playing")
                audio.stop()
                exit()
        c = c + 1