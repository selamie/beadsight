import pyrealsense2 as rs
import cv2
import numpy as np
from typing import List, Tuple, Dict
from multiprocessed_cameras import SaveVideosMultiprocessed
from beadsight_realtime import BeadSight
import time

CAM_SERIALS = {
        1: '220222066259',
        2: '151322066099',
        3: '151322069488',
        4: '151322061880',
        5: '151322066932',
        6: '152522250441'
    }


class CamerasAndBeadSight:
    def __init__(self, 
                 cameras:List[int] = [1,2,3,4,5,6],
                 shapes:List[Tuple[int, int]] = [(1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (800, 1280)],
                 frame_rate:int = 30) -> None:
        
        self.pipelines = []
        configs = []
        self.aligned_streams = []
        for i, cam in enumerate(cameras):
            print('starting camera', cam)
            H, W = shapes[i]
            self.pipelines.append(rs.pipeline())
            configs.append(rs.config())
            configs[-1].enable_device(CAM_SERIALS[cam])
            configs[-1].enable_stream(rs.stream.color, W, H, rs.format.bgr8, frame_rate)
            self.pipelines[-1].start(configs[-1])
            self.aligned_streams.append(rs.align(rs.stream.color))      

        self.frame_num = 0 # keep track of the frame number
        self.cameras = cameras
        self.beadsight = BeadSight()

    def get_next_frames(self) -> Dict[str, np.ndarray]:
        all_frames = {}
        for i, cam in enumerate(self.cameras):
            frames = self.pipelines[i].wait_for_frames()
            frames = self.aligned_streams[i].process(frames)
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            all_frames[str(cam)] = color_image
        r, im = self.beadsight.get_frame()
        all_frames["beadsight"] = im

        return all_frames
    
    def close(self) -> None:
        self.beadsight.close()

        


if __name__ == "__main__":
    
    video_paths = ["data/ssd/cam1.avi", "data/ssd/cam2.avi","data/ssd/cam3.avi","data/ssd/cam4.avi","data/ssd/cam5.avi","data/ssd/cam6.avi","data/ssd/beadsight.avi"]
    camera_sizes = [(1080, 1920), (1080, 1920),(1080, 1920), (1080, 1920),(1080, 1920),(800,1280), (480,480)]

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'HFYU')
    save_videos = SaveVideosMultiprocessed(video_paths, camera_sizes, fourcc, fps)

    cameras = CamerasAndBeadSight()

    denom = 0 
    avg = 0
    while True:
        start = time.time()
        frames = cameras.get_frames()
        save_videos(list(frames.values())) 
        print("time:", time.time()-start)
        denom+=1
        avg = (avg +(time.time()-start))
        print("avg",avg/denom)   #running avg
