import pyrealsense2 as rs
import cv2
import numpy as np
import time
from multiprocessing import Process, Queue, Array, Pool
import ctypes
from typing import List, Tuple, Dict

CAM_SERIALS = {
        1: '220222066259',
        2: '151322066099',
        3: '151322069488',
        4: '151322061880',
        5: '151322066932',
        6: '152522250441'
    }

class MultiprocessedCameras:
    def __init__(self, 
                 cameras:List[int] = [1,2,3,4,5],
                 shapes:List[Tuple[int, int]] = [(480, 848)] * 5,
                 frame_rate:int = 30):
        
        self.cameras = cameras
        self.sizes = shapes

        memory_size = 0
        for shape in shapes:
            memory_size += shape[0]*shape[1]*3

        # initialize the shared memory
        self.image_array = Array(ctypes.c_uint8, memory_size, lock=True)

        # create a queue to store the received frame numbers
        self.frame_num_queue = Queue()
        self.process = Process(target=self._run_cameras, 
                               daemon=True, 
                               args=(self.image_array, 
                                     self.frame_num_queue, 
                                     self.cameras, 
                                     self.sizes, 
                                     frame_rate))
        self.process.start()

        # wait for the first frame to be received to initialize the shared memory
        self.last_received_frame_num = self.frame_num_queue.get(block=True)

    @staticmethod    
    def _run_cameras(image_array,
                     frame_num_queue:Queue, 
                     cameras:List[int], 
                     sizes:List[Tuple[int, int]],
                     frame_rate:int = 30):
        # initialize the cameras
        pipelines = []
        configs = []
        aligned_streams = []
        for i, cam in enumerate(cameras):
            print('starting camera', cam)
            H, W = sizes[i]
            pipelines.append(rs.pipeline())
            configs.append(rs.config())
            configs[-1].enable_device(CAM_SERIALS[cam])
            configs[-1].enable_stream(rs.stream.color, W, H, rs.format.bgr8, frame_rate)
            pipelines[-1].start(configs[-1])
            aligned_streams.append(rs.align(rs.stream.color))      

        frame_num = 0 # keep track of the frame number

        # View shared memory as a numpy array so that we can edit it in place
        # create a numpy array from the shared memory for each camera
        image_array_np = np.ctypeslib.as_array(image_array.get_obj())

        while True:
            # get the frames
            running_index = 0
            raw_frames = []
            for i, cam in enumerate(cameras):
                H, W = sizes[i]
                frames = pipelines[i].wait_for_frames()
                frames = aligned_streams[i].process(frames)
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                raw_frames.append(color_image)

                # save the image to disk:
                # start_time = time.time()
                # np.save(f"ssd/raw_images/camera_{cam}_frame_{frame_num}.npy", color_image)
                # print("Time to save image: ", time.time() - start_time)

            # update the shared memory
            running_index = 0
            with image_array.get_lock():
                for i, frame in enumerate(raw_frames):
                    H, W = sizes[i]
                    image_array_np[running_index:running_index + H*W*3] = frame.flatten()
                    running_index += H*W*3
            frame_num_queue.put(frame_num) # let the main process know that a new frame is ready
            frame_num += 1

            

    def get_frames(self):
        with self.image_array.get_lock():
            image_array_np = np.ctypeslib.as_array(self.image_array.get_obj())
            images = {}
            running_index = 0
            for i, cam in enumerate(self.cameras):
                H, W = self.sizes[i]
                images[str(cam)] = image_array_np[running_index:running_index + H*W*3].reshape((H, W, 3)).copy()
                running_index += H*W*3

        return images
    
    def get_next_frames(self):
        # wait until a new frame is ready
        current_frame_num = self.frame_num_queue.get(block=True)
        assert current_frame_num > self.last_received_frame_num, "Received a frame that was already received"

        # Empty the queue
        while True:
            try:
                current_frame_num = self.frame_num_queue.get(block=False)
            except:
                break

        # update the last received frame number
        # if self.last_received_frame_num + 1 != current_frame_num:
            # print("Missed frames: ", self.last_received_frame_num, current_frame_num)
        self.last_received_frame_num = current_frame_num

        # return the frames
        return self.get_frames()
    
    def close(self):
        self.process.terminate()
        self.process.join()
    
    def __del__(self):
        self.close()

class Cameras:
    def __init__(self, 
                 cameras:List[int] = [1,2,3,4,5],
                 shapes:List[Tuple[int, int]] = [(480, 848)] * 5,
                 frame_rate:int = 30):
        
        self.cameras = cameras
        self.sizes = shapes

        # initialize the cameras
        self.pipelines = []
        self.configs = []
        self.aligned_streams = []
        for i, cam in enumerate(cameras):
            print('starting camera', cam)
            H, W = shapes[i]
            self.pipelines.append(rs.pipeline())
            self.configs.append(rs.config())
            self.configs[-1].enable_device(CAM_SERIALS[cam])
            self.configs[-1].enable_stream(rs.stream.color, W, H, rs.format.bgr8, frame_rate)
            self.pipelines[-1].start(self.configs[-1])
            self.aligned_streams.append(rs.align(rs.stream.color))      
        self.closed = False

    def get_frames(self):
        frames = {}
        for i, cam in enumerate(self.cameras):
            frame = self.pipelines[i].wait_for_frames()
            frame = self.aligned_streams[i].process(frame)
            color_frame = frame.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            frames[str(cam)] = color_image
        return frames
    
    def get_next_frames(self):
        return self.get_frames()
    
    def close(self):
        for pipeline in self.pipelines:
            pipeline.stop()
        self.closed = True
    
    def __del__(self):
        if not self.closed:
            self.close()
        
class SaveImagesMultiprocessedOld():
    def __init__(self, 
                 sizes:List[Tuple[int, int]],
                 n_workers:int,
                 encoding_params:List[int] = None):
        
        self.n_workers: int = n_workers
        self.sizes:List[Tuple[int, int]] = sizes
        self.frame_number: int = 0

        total_size: int = 0
        for size in sizes:
            total_size += size[0]*size[1]*3

        self.shared_memory_arrays = []
        self.frame_info_queues: List[Queue] = []
        self.workers: List[Process] = []
        for _ in range(n_workers):
            shared_memory = Array(ctypes.c_uint8, total_size, lock=True)
            frame_info_queue = Queue()
            worker = Process(target=self._run, 
                             args=(shared_memory, 
                                   frame_info_queue,
                                   sizes,
                                   encoding_params))
            worker.start()
            self.workers.append(worker)
            self.shared_memory_arrays.append(shared_memory)
            self.frame_info_queues.append(frame_info_queue)
    

    @staticmethod
    def _run(image_data, 
             frame_info_queue:Queue, 
             sizes:List[Tuple[int, int]], 
             encoding_params:List[int]):
        
        image_data_np = np.ctypeslib.as_array(image_data.get_obj())
        while True:
            frame_num, save_paths = frame_info_queue.get() # get the frame info, waits until a new frame is ready

            if frame_num == -1: # if the frame number is -1, then we are done
                break

            with image_data.get_lock():
                idx = 0 # keep track of the index in the shared memory
                for i, save_path in enumerate(save_paths):
                    H, W = sizes[i]
                    if encoding_params is None:
                        cv2.imwrite(save_path, image_data_np[idx:idx + H*W*3].reshape((H, W, 3)))
                    else:
                        cv2.imwrite(save_path, image_data_np[idx:idx + H*W*3].reshape((H, W, 3)), 
                                    encoding_params)
                    idx += H*W*3

    def save_images(self, images:List[np.ndarray], save_paths:List[str]):
        assert len(images) == len(save_paths), "Number of images and save paths must be the same"
        worker_index = self.frame_number % self.n_workers # rotate through the workers
        
        # save the images to the shared memory
        with self.shared_memory_arrays[worker_index].get_lock():
            # create a numpy array from the shared memory to make it easier to save the images
            image_data_np = np.ctypeslib.as_array(self.shared_memory_arrays[worker_index].get_obj())
            running_index = 0 # keep track of the index in the shared memory
            for i, image in enumerate(images):
                H, W = self.sizes[i]
                image_data_np[running_index:running_index + H*W*3] = image.ravel()
                running_index += H*W*3
        
        # put the frame info in the queue
        self.frame_info_queues[worker_index].put((self.frame_number, save_paths))
        self.frame_number += 1
    
    def close(self):
        for queue in self.frame_info_queues:
            queue.put((-1, []))
        for worker in self.workers:
            worker.join()
    
    def __del__(self):
        self.close()

from multiprocessing import shared_memory
class SaveImagesMultiprocessed():
    def __init__(self, 
                 sizes:List[Tuple[int, int]],
                 n_workers:int,
                 buffer_size:int = 100,
                 encoding_params:List[int] = None):
        
        self.n_workers: int = n_workers
        self.sizes:List[Tuple[int, int]] = sizes
        self.frame_number: int = 0

        total_size: int = 0
        for size in sizes:
            total_size += size[0]*size[1]*3

        self.available_queue: Queue = Queue()
        for i in range(buffer_size):
            self.available_queue.put(i)
        self.frame_queue: Queue = Queue()
        self.workers: List[Process] = []
        self.shared_memory_arrays = [Array(ctypes.c_uint8, total_size, lock=True) for _ in range(buffer_size)]
        for _ in range(n_workers):
            worker = Process(target=self._run, 
                             args=(self.frame_queue,
                                   self.available_queue,
                                   self.shared_memory_arrays,
                                   sizes,
                                   encoding_params))
            worker.start()
            self.workers.append(worker)
    

    @staticmethod
    def _run(frame_queue:Queue, 
             available_queue:Queue,
             shared_memory_arrays,
             sizes:List[Tuple[int, int]], 
             encoding_params:List[int]):
        
        tot_size = sum([H*W*3 for H, W in sizes])
        
        while True:
            frame_num, array_idx, save_paths = frame_queue.get() # get the frame info, waits until a new frame is ready

            if frame_num == -1: # if the frame number is -1, then we are done
                break
            
            with shared_memory_arrays[array_idx].get_lock():
                image_data_np = np.ctypeslib.as_array(shared_memory_arrays[array_idx].get_obj())
                idx = 0 # keep track of the index in the shared memory
                for i, save_path in enumerate(save_paths):
                    H, W = sizes[i]
                    if encoding_params is None:
                        cv2.imwrite(save_path, image_data_np[idx:idx + H*W*3].reshape((H, W, 3)))
                    else:
                        cv2.imwrite(save_path, image_data_np[idx:idx + H*W*3].reshape((H, W, 3)), 
                                    encoding_params)
                    idx += H*W*3
            available_queue.put(array_idx)

    def __call__(self, images:List[np.ndarray], save_paths:List[str]):
        assert len(images) == len(save_paths), "Number of images and save paths must be the same"
        
        # get the next available shared memory
        # print("Available queue size: ", self.available_queue.qsize())
        array_idx = self.available_queue.get()
        with self.shared_memory_arrays[array_idx].get_lock():
            image_data_np = np.ctypeslib.as_array(self.shared_memory_arrays[array_idx].get_obj())
            running_index = 0 # keep track of the index in the shared memory
            for i, image in enumerate(images):
                H, W = self.sizes[i]
                image_data_np[running_index:running_index + H*W*3] = image.ravel()
                running_index += H*W*3

        self.frame_queue.put((self.frame_number, array_idx, save_paths))
        self.frame_number += 1
    
    def close(self):
        # for queue in self.frame_info_queues:
        #     queue.put((-1, []))
        for i in range(self.n_workers):
            self.frame_queue.put((-1, None, []))
        for worker in self.workers:
            worker.join()
    
    def __del__(self):
        self.close()
    
class SaveVideosMultiprocessed():
    def __init__(self, 
                 file_names:List[str],
                 sizes:List[Tuple[int, int]],
                 fourcc_encoder = None,
                 fps = 30):
        
        self.sizes:List[Tuple[int, int]] = sizes
        self.frame_number: int = 0
        
        self.shared_memory_arrays = []
        self.frame_info_queues: List[Queue] = []
        self.workers: List[Process] = []
        for i, file_name in enumerate(file_names):
            image_size = sizes[i][0]*sizes[i][1]*3
            shared_memory = Array(ctypes.c_uint8, image_size, lock=True)
            frame_info_queue = Queue()
            worker = Process(target=self._run, 
                             args=(file_name,
                                   shared_memory, 
                                   frame_info_queue,
                                   (sizes[i][0], sizes[i][1], 3),
                                   fourcc_encoder,
                                   fps))
            
            worker.start()
            self.workers.append(worker)
            self.shared_memory_arrays.append(shared_memory)
            self.frame_info_queues.append(frame_info_queue)
    

    @staticmethod
    def _run(file_name,
             image_data, 
             frame_info_queue:Queue, 
             image_size:Tuple[int, int, int], # (H, W, C)
             fourcc_encoder,
             fps:int):
        
        last_frame = -1
        video_writer = cv2.VideoWriter(file_name, fourcc_encoder, fps, (image_size[1], image_size[0]), isColor=True)
        image_data_np = np.ctypeslib.as_array(image_data.get_obj())
        while True:
            frame_num = frame_info_queue.get() # get the frame info, waits until a new frame is ready
            if frame_num == -1: # if the frame number is -1, then we are done
                break
                
            assert frame_num == last_frame + 1, "Missed frames"
            last_frame = frame_num

            with image_data.get_lock():
                video_writer.write(image_data_np.reshape(image_size))

        video_writer.release()

    def __call__(self, images:List[np.ndarray]):
       
        for i, image in enumerate(images):
            # save the images to the shared memory
            with self.shared_memory_arrays[i].get_lock():
                # create a numpy array from the shared memory to make it easier to save the images
                image_data_np = np.ctypeslib.as_array(self.shared_memory_arrays[i].get_obj())
                image_data_np[:] = image.ravel()
        
            # put the frame info in the queue
            self.frame_info_queues[i].put(self.frame_number)
        
        self.frame_number += 1
    
    def close(self):
        for queue in self.frame_info_queues:
            queue.put(-1)
        for worker in self.workers:
            worker.join()
    
    def __del__(self):
        self.close()

if __name__ == "__main__":
    overall_time = time.time()
    # test the class
    n_frames = 1000
    cams = [1, 2, 3, 4, 5, 6]
    sizes = [(1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (800, 1280)]

    multi_cam = MultiprocessedCameras(cams, sizes, frame_rate=30)
    # multi_save = SaveImagesMultiprocessed(sizes, 12, buffer_size=100, encoding_params=[int(cv2.IMWRITE_JPEG_QUALITY), 95])

    video_names = ["ssd/test1.avi", "ssd/test2.avi", "ssd/test3.avi", "ssd/test4.avi", "ssd/test5.avi", "ssd/test6.avi"]
    frame_rate = 30
    width = 1920
    height = 1080
    fourcc = cv2.VideoWriter_fourcc(*'HFYU')
    # compression_level = 3  # Adjust compression level as needed
    video_save = SaveVideosMultiprocessed(video_names, sizes, fourcc, frame_rate)
    # video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height), isColor=True)#, params=[cv2.IMWRITE_FFMPEG_CODEC_QUALITY, compression_level])

    for i_frame in range(n_frames):
        start_time = time.time()
        frames: Dict[str, np.ndarray] = multi_cam.get_next_frames()
        save_names = [f"ssd/images/camera_{cam}_frame_{i_frame}.jpg" for cam in frames]
        # save_start_time = time.time()
        # video.write(frames['1'])
        # video_save(list(frames.values()))
        # multi_save(list(frames.values()), save_names)
        # print("Time to save images: ", time.time() - save_start_time)
        # print("Time to get frames: ", time.time() - start_time)

        # render the images with cv2
        for cam, frame in frames.items():
            cv2.imshow(f"camera {cam}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    # multi_save.close()
    # video.release()
    print('start closing')
    video_save.close()
    print("overall time: ", time.time() - overall_time)

    exit()