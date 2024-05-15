import os
import cv2
import time
import argparse
from defisheye import Defisheye

vkwargs = {"fov": 180,
            "pfov": 120,
            "xcenter": None,
            "ycenter": None,
            "radius": None,
            "angle": 0,
            "dtype": "linear",
            "format": "fullframe"
            }

for im in range(1,4):
    folder = "/media/selamg/DATA/Tactile_Sensor/Unwarp_Fisheye/Fisheye_Imgs"
    input_image = folder + '/test' + str(im) + '.jpg'
    print(input_image)

    if im == 1:
        og = cv2.imread(input_image)
        start = time.time()
        obj = Defisheye(input_image, **vkwargs)
        x,y,i,j = obj.calculate_conversions()
        end_class = time.time()
        unwarped = obj.unwarp(og)
        end_warp = time.time()

        print("\nInstantiate Class Time: ", end_class - start)
        print("\nWarp Time: ", end_warp - end_class)

    else:
        # input_image = 'Fisheye_Imgs/test' + str(i) + '.jpg'
        og = cv2.imread(input_image)
        start = time.time()
        unwarped = obj.unwarp(og)
        end_warp = time.time()

        print("\nWarp Time: ", end_warp - start)

    cv2.imshow("original"+str(im), og)
    cv2.imshow("undistorted", unwarped)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # cv2.imshow("undistorted", unwarped)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()