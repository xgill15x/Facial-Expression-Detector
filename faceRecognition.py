# Author: Jason Gill

import cv2
import helpers

if __name__ == '__main__':

    # open img
    myImage = helpers.readImage()

    # find faces
    facesInMyImage = helpers.detectFace(myImage)

    # detects emotions and draws rectangles on them
    allCroppedImageURLs = []
    helpers.detectEmotion(facesInMyImage, allCroppedImageURLs, myImage)

    # show faces
    cv2.imshow("Faces Found", myImage)

    # destroy cropped images during run time
    helpers.destroyAllTempImages(allCroppedImageURLs)
    helpers.endScene()


