
#imports
import numpy as np
import face_recognition
import os
from PIL import Image , ImageDraw ,ExifTags
import datetime
import dlib
import pymongo

dlib.DLIB_USE_CUDA = False #Turn off GPU Usage

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
  # Create a ImageDraw instance
  draw = ImageDraw.Draw(pil_image)
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
                    draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

      # Draw label
      text_width, text_height = draw.textsize(name)
      draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
      draw.text((left + 6, bottom - text_height - 5), name +" "+ str(np.min(face_distances)) , fill=(0,0,0))
    else:
      if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

      # Draw label
      text_width, text_height = draw.textsize(name)
      draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
      draw.text((left + 6, bottom - text_height - 5), "Unknown" +" "+ str(np.min(face_distances))[0:4] , fill=(0,0,0))

        

  del draw
  rgb_im = pil_image.convert('RGB')

  # Save image
  rgb_im.save("./Names/"+pic)
  i = i+1
  print("Image Saved no. "+str(i))
