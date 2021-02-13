
#imports
import numpy as np
import face_recognition
import os
from PIL import Image ,ExifTags
import datetime
import dlib
import pymongo
from config import mongo_url

dlib.DLIB_USE_CUDA = False #Turn off GPU Usage
import pymongo
import urllib
myclient = pymongo.MongoClient(mongo_url)
mydb = myclient["FaceRec"]
print("Connected to DB successfully ....")

mycol = mydb["Faces"]
print("Created Collection successfully ....")

uk_images = os.listdir("./unknown")
known_images = os.listdir("./known")

known_face_encodings = []
face_encoding = None
for image in known_images:
    k_image = face_recognition.load_image_file("./known/"+image)
    face_encoding = face_recognition.face_encodings(k_image)[0]
    known_face_encodings.append(face_encoding)

known_face_names = []
for image in known_images:
    known_face_names.append(image[:-4])

i = 0
names = []
face_distances1 = []
name = ""
acc = 0.45 # Decrease the accuracy to decrease false positive rate and vice versa
for pic in uk_images:
  img_path = './unknown/'+pic
  pil_image = Image.open(img_path).convert("RGB")
  img_exif = pil_image.getexif()
  ret = {}
  orientation  = 0
  if img_exif:
      for tag, value in img_exif.items():
          decoded = ExifTags.TAGS.get(tag, tag)
          ret[decoded] = value
      orientation  = ret["Orientation"]
  if orientation == 8:
      pil_image = pil_image.rotate(90, Image.NEAREST, expand=1)
  elif orientation == 3:
      pil_image = pil_image.rotate(180, Image.NEAREST, expand=1)
  elif orientation == 6:
      pil_image = pil_image.rotate(270, Image.NEAREST, expand=1)
  test_image = np.asarray(pil_image)

  face_locations = face_recognition.face_locations(test_image)
  face_encodings = face_recognition.face_encodings(test_image, face_locations)
  names = []
  face_distances1 = []

  for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    face_distances1.append(face_distances)
    best_match_index = np.argmin(face_distances)
    if(np.min(face_distances) < acc ):
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
        x = mycol.count_documents({"name" : name})
        if (x != 0 ):
            mycol.update_one(
                {"name" : name},
                { "$addToSet" : {"images" : pic }}
            )
        else:
            mydict = { "name": name, "images": [pic] }
            x = mycol.insert_one(mydict)
    else:
        x = mycol.count_documents({"name" : "Unknown"})
        if (x!= 0 ):
            mycol.update_one(
                {"name" : "Unknown"},
                { "$addToSet" : {"images" : pic }}
            )
        else:
            mydict = { "name": "Unknown", "images": [pic] }
            x = mycol.insert_one(mydict)
  i = i+1
  print("Image Saved no. "+str(i))
