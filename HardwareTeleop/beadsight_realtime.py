#!/usr/bin/env python

import cv2
import time
from defisheye import Defisheye 
# from HardwareTeleop.defisheye import Defisheye 
#this won't work if you try to run this file but works for the relative import

# DEVICENUM = 0 # beadsight is at /dev/video[devicenum]
#!/usr/bin/env python3.6
# -*- Coding: UTF-8 -*-
from numpy import arange, sqrt, arctan, sin, tan, meshgrid, pi
from numpy import ndarray, hypot


class Defisheye:
    """
    Defisheye
    fov: fisheye field of view (aperture) in degrees
    pfov: perspective field of view (aperture) in degrees
    xcenter: x center of fisheye area
    ycenter: y center of fisheye area
    radius: radius of fisheye area
    angle: image rotation in degrees clockwise
    dtype: linear, equalarea, orthographic, stereographic
    format: circular, fullframe
    """

    def __init__(self, infile, **kwargs):
        vkwargs = {"fov": 180,
                   "pfov": 120,
                   "xcenter": None,
                   "ycenter": None,
                   "radius": None,
                   "angle": 0,
                   "dtype": "equalarea",
                   "format": "fullframe"
                   }
        self._start_att(vkwargs, kwargs)

        if type(infile) == str:
            _image = cv2.imread(infile)
        elif type(infile) == ndarray:
            _image = infile
        else:
            raise Exception("Image format not recognized")

        width = _image.shape[1]
        height = _image.shape[0]
        xcenter = width // 2
        ycenter = height // 2

        dim = min(width, height)
        self.x0 = xcenter - dim // 2
        self.xf = xcenter + dim // 2
        self.y0 = ycenter - dim // 2
        self.yf = ycenter + dim // 2

        self._image = _image[self.y0:self.yf, self.x0:self.xf, :]

        self._width = self._image.shape[1]
        self._height = self._image.shape[0]

        if self._xcenter is None:
            self._xcenter = (self._width - 1) // 2

        if self._ycenter is None:
            self._ycenter = (self._height - 1) // 2

    def _map(self, i, j, ofocinv, dim):

        xd = i - self._xcenter
        yd = j - self._ycenter

        rd = hypot(xd, yd)
        phiang = arctan(ofocinv * rd)

        if self._dtype == "linear":
            ifoc = dim * 180 / (self._fov * pi)
            rr = ifoc * phiang
            # rr = "rr={}*phiang;".format(ifoc)

        elif self._dtype == "equalarea":
            ifoc = dim / (2.0 * sin(self._fov * pi / 720))
            rr = ifoc * sin(phiang / 2)
            # rr = "rr={}*sin(phiang/2);".format(ifoc)

        elif self._dtype == "orthographic":
            ifoc = dim / (2.0 * sin(self._fov * pi / 360))
            rr = ifoc * sin(phiang)
            # rr="rr={}*sin(phiang);".format(ifoc)

        elif self._dtype == "stereographic":
            ifoc = dim / (2.0 * tan(self._fov * pi / 720))
            rr = ifoc * tan(phiang / 2)

        rdmask = rd != 0
        xs = xd.copy()
        ys = yd.copy()

        xs[rdmask] = (rr[rdmask] / rd[rdmask]) * xd[rdmask] + self._xcenter
        ys[rdmask] = (rr[rdmask] / rd[rdmask]) * yd[rdmask] + self._ycenter

        xs[~rdmask] = 0
        ys[~rdmask] = 0

        xs = xs.astype(int)
        ys = ys.astype(int)
        return xs, ys
    
    def calculate_conversions(self):
        """
        Added functionality to allow for a single calculated mapping to be applied to a series of images
        from the same fisheye camera.
        """
        if self._format == "circular":
            dim = min(self._width, self._height)
        elif self._format == "fullframe":
            dim = sqrt(self._width ** 2.0 + self._height ** 2.0)

        if self._radius is not None:
            dim = 2 * self._radius

        # compute output (perspective) focal length and its inverse from ofov
        # phi=fov/2; r=N/2
        # r/f=tan(phi);
        # f=r/tan(phi);
        # f= (N/2)/tan((fov/2)*(pi/180)) = N/(2*tan(fov*pi/360))

        ofoc = dim / (2 * tan(self._pfov * pi / 360))
        ofocinv = 1.0 / ofoc

        i = arange(self._width)
        j = arange(self._height)
        self.i, self.j = meshgrid(i, j)

        self.xs, self.ys, = self._map(self.i, self.j, ofocinv, dim)
        return self.xs, self.ys, self.i, self.j
    
    def unwarp(self, image):
        """
        Added functionality to allow for a single calculated mapping to be applied to a series of images
        from the same fisheye camera.
        """
        image = image[self.y0:self.yf, self.x0:self.xf, :]
        img = image.copy()
        img[self.i, self.j, :] = image[self.xs, self.ys, :]
        return img

    def convert(self, outfile=None):
        if self._format == "circular":
            dim = min(self._width, self._height)
        elif self._format == "fullframe":
            dim = sqrt(self._width ** 2.0 + self._height ** 2.0)

        if self._radius is not None:
            dim = 2 * self._radius

        # compute output (perspective) focal length and its inverse from ofov
        # phi=fov/2; r=N/2
        # r/f=tan(phi);
        # f=r/tan(phi);
        # f= (N/2)/tan((fov/2)*(pi/180)) = N/(2*tan(fov*pi/360))

        ofoc = dim / (2 * tan(self._pfov * pi / 360))
        ofocinv = 1.0 / ofoc

        i = arange(self._width)
        j = arange(self._height)
        i, j = meshgrid(i, j)

        xs, ys, = self._map(i, j, ofocinv, dim)
        img = self._image.copy()

        img[i, j, :] = self._image[xs, ys, :]
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img

    def _start_att(self, vkwargs, kwargs):
        """
        Starting atributes
        """
        pin = []

        for key, value in kwargs.items():
            if key not in vkwargs:
                raise NameError("Invalid key {}".format(key))
            else:
                pin.append(key)
                setattr(self, "_{}".format(key), value)

        pin = set(pin)
        rkeys = set(vkwargs.keys()) - pin
        for key in rkeys:
            setattr(self, "_{}".format(key), vkwargs[key])

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

    beadcam = BeadSight(6)
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc_mp4, 30.0, (480, 480), isColor=True) 
    
    while(True):

        r, im = beadcam.get_frame()

        if not r:
            break
            
        out.write(im)

        cv2.imshow('im',im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    beadcam.cap.release()
    cv2.destroyAllWindows()