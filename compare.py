import cv2
import numpy as np
import face_recognition

imgVijay = face_recognition.load_image_file('imagesbasic/spb.jpg')
imgVijay = cv2.cvtColor(imgVijay,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imagesbasic/ravi.jpeg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgVijay)[0]
encodeVijay = face_recognition.face_encodings(imgVijay)[0]
cv2.rectangle(imgVijay,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeVijay],encodeTest)
faceDis = face_recognition.face_distance([encodeVijay],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
#print(faceLoc)
cv2.imshow('vijay',imgVijay)
cv2.imshow('ravi',imgTest)
cv2.waitKey()
