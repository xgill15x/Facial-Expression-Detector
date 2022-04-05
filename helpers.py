# Author: Jason Gill

import sys, os
import cv2
from PIL import Image
from deepface import DeepFace

# destroys all cropped images created during runtime
def destroyAllTempImages(allCroppedImageURLs):
    for (croppedImage) in allCroppedImageURLs:
        os.remove(croppedImage)

# waits for user to press a key to end program
def endScene():
    print("Press any key on detection window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# reads given image
def readImage():
    imagePath = sys.argv[1]
    myImage = cv2.imread(imagePath)

    return myImage

# detects faces in image, stores them in an array
def detectFace(myImage):
    myImageGreyScale = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)

    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    facesInMyImage = faceCascade.detectMultiScale(
        myImageGreyScale,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    return facesInMyImage

# detects emotion of given images and draws rectangles around them
def detectEmotion(facesInMyImage, allCroppedImageURLs, myImage):
    for (x, y, w, h) in facesInMyImage:
        # Open the image with PIL library
        PILVersionOfImage = Image.open(sys.argv[1])

        # crop the faces from the image
        myCroppedFace = PILVersionOfImage.crop((x, y, x + w, y + h))

        # saving cropped face to file
        uniqueCroppedImageURL = "croppedImageDir/croppedFace" + str(x) + ".jpg"
        myCroppedFace.save(uniqueCroppedImageURL)

        # adding unique cropped image url to global list of cropped URLs
        allCroppedImageURLs.append(uniqueCroppedImageURL)

        # drawing rectangles over detected faces
        detectionRectangle = cv2.rectangle(myImage,
                                           (x, y),
                                           (x + w, x + y),
                                           (0, 255, 0),
                                           3)

        # analyze emotion
        for (croppedImage) in allCroppedImageURLs:
            faceAnalysis = DeepFace.analyze(croppedImage, enforce_detection=False)
            print(faceAnalysis)

        # write event details
        font = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText = (x , y-10)
        fontScale = 1
        fontColor = (0, 255, 0)
        thickness = 1
        lineType = 2

        # writes emotion on gui
        cv2.putText(detectionRectangle, str(faceAnalysis['dominant_emotion']),
                    topLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)