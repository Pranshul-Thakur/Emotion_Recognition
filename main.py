import cv2
import Simple_facerec from SimpleFacerec
import face_recognition

#encoding the faces
sfr = SimpleFacerec()
sfr.load_encoding_images("C:\Users\LENOVO\Desktop\Face-Detect\images") #location of images

# Load Camera
cap = cv2.VideoCapture(2)


while True:
    ret, frame = cap.read()
    face_locations, face_names = sfr.detect_known_faces(frame) #detect the faces
    for face_loc, name in zip(face_locations, face_names):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3] #mapping the points on the face
        
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) #formation of the square and displaying of text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()

""" 
This one is for stationary images

"""

"""
img = cv2.imread("Astuti.png") #path of image
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("ronaldo")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result:", result)

cv2.imshow("Img", img)
cv2.imshow("Img2", img2)
cv2.waitKey(0)

"""
