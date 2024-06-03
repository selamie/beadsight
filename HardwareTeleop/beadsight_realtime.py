#!/usr/bin/env python

import cv2
import time
from HardwareTeleop.defisheye import Defisheye 
#this won't work if you try to run this file but works for the relative import

# DEVICENUM = 0 # beadsight is at /dev/video[devicenum]

class BeadSight():

    def __init__(self, DEVICENUM,FRAME_WIDTH=640,FRAME_HEIGHT=512):
        vkwargs = {"fov": 180,
            "pfov": 120,
            "xcenter": None,
            "ycenter": None,
            "radius": None,
            "angle": 0,
            "dtype": "linear",
            "format": "fullframe"
            }

        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        self.cap = cv2.VideoCapture(DEVICENUM) 

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = self.cap.read()
        print(ret)
        if not ret: 
            raise ValueError("cap.read failed, device not plugged in or wrong device number selected")
        self.defish = Defisheye(frame,**vkwargs)
        x,y,i,j = self.defish.calculate_conversions()
        unwarped = self.defish.unwarp(frame)

    def get_frame(self):
        ret,frame = self.cap.read()

        unwarped = self.defish.unwarp(frame)

        return ret, unwarped
    
    def close(self):
        self.cap.release()

def test1():
    
    beadcam = BeadSight()
    while(True):
        time.sleep(0.1) 
        start_time = time.time()
        r,im = beadcam.get_frame()
        print(time.time()-start_time)
        # input()        
        # without sleep--0.035
        # with sleep--0.0175 (so prob fine?)

if __name__ == '__main__':

    # test1()

    beadcam = BeadSight(0)
    while(True):
        r, im = beadcam.get_frame()
        if r:
            # cv2.imshow('og',og)
            cv2.imshow('unwarped',im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    beadcam.cap.release()
    cv2.destroyAllWindows()