# Face-Recognition

# Requirements
1. Python3
2. face-recognition (https://pypi.org/project/face-recognition/)
3. Numpy
4. pymongo

# Files
1. Get_Faces.py 
Gets faces from the images and stores in a known folder (Create a folder "known" before running this code )

2. Face_reg.py
Matches the faces from the "known" folder to all the images in "unknown" folder and recreate images with people tagged and stores in "Names" folder

3. Face_reg_mongo.py
Matches the faces from the "known" folder to all the images in "unknown" folder and stores the image name with the persons name in MongoDB
