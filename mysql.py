import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import mysql.connector
#import warnings
#warnings.filterwarnings("ignores")

path = 'imagesattendance'
images = []
classNames = []
myList = os.listdir(path)
print("list", myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        #print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            #nameList.append(entry[0])
            #print(nameList)
        if name not in nameList:
            #print("i am here")
            now = datetime.now()
            dtString = now.strftime('%d-%m-%y,%H:%M:%S')
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="attendance")

            mycursor = mydb.cursor()

            sql = "INSERT INTO face_attendance (name, datetime) VALUES (%s, %s);"
            val = (name, str(datetime.now()))
            mycursor.execute(sql, val)

            mydb.commit()

            #print("record inserted ",mycursor)
            print(mycursor.rowcount, "record inserted.")
            now = datetime.now()
            dtString = now.strftime('%d-%m-%y,%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            #print("CSV successfully updated")
            entry = line.split(',')
            nameList.append(entry[0])


encodeListknown = findEncodings(images)
# print(len(encodeListknown))
print('Encoding Complete')

cap = cv2.VideoCapture(0)
#print(cap, "is open:", cap.isOpened())

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListknown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
        else:
            print("Not Found")
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Not Found", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()


