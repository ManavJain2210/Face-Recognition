import numpy as np
import face_recognition
import os
from PIL import Image , ImageDraw ,ExifTags
import datetime
import dlib

dlib.DLIB_USE_CUDA = False
i = 0 
images = ["List of Images to extract faces from"]
for pic in images:
    image = face_recognition.load_image_file("./unknown/" + pic)
    face_loc = face_recognition.face_locations(image)

    for face_location in face_loc:
        top, right, bottom, left = face_location

        face_image = image[top :bottom , left :right]
        pil_image = Image.fromarray(face_image)
        # pil_image.show()
        x = datetime.datetime.now()
        pil_image.save(f'./known/Img_{str(x.time())}.jpg')
        i = i+1
        print("Added face no "+str(i))