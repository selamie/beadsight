import cv2
import time
from defisheye import Defisheye


class BeadSight():

    def __init__(self, FRAME_WIDTH=640,FRAME_HEIGHT=512):
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
        self.cap = cv2.VideoCapture(0) 

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = self.cap.read()
        self.defish = Defisheye(frame,**vkwargs)
        x,y,i,j = self.defish.calculate_conversions()
        unwarped = self.defish.unwarp(frame)

    def get_frame(self):
        ret,frame = self.cap.read()

        unwarped = self.defish.unwarp(frame)

        return ret, unwarped, frame

def test1():
    
    beadcam = BeadSight()
    while(True):
        time.sleep(0.1) 
        start_time = time.time()
        r,im,og = beadcam.get_frame()
        print(time.time()-start_time)
        # input()        
        # without sleep--0.035
        # with sleep--0.0175 (so prob fine?)

if __name__ == '__main__':

    # test1()

    beadcam = BeadSight()
    while(True):
        r, im, og = beadcam.get_frame()
        if r:
            # cv2.imshow('og',og)
            cv2.imshow('unwarped',im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    beadcam.cap.release()
    cv2.destroyAllWindows()