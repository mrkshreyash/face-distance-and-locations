import face_recognition

img = face_recognition.load_image_file("Test.jpg")
locations = face_recognition.face_locations(img)

landmark = face_recognition.face_landmarks(img, locations, model="small")

print(landmark[0].keys())


