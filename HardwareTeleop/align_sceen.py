import cv2
import numpy as np
from multiprocessed_cameras import MultiprocessedCameras

# Define the paths to the video files for each camera
video_paths = [
    # "/home/selamg/beadsight/data/stonehenge_examples/episode_0/cam-1.avi",
    # "/home/selamg/beadsight/data/stonehenge_examples/episode_0/cam-2.avi",
    # "/home/selamg/beadsight/data/stonehenge_examples/episode_0/cam-3.avi",
    # "/home/selamg/beadsight/data/stonehenge_examples/episode_0/cam-4.avi",
    # "/home/selamg/beadsight/data/stonehenge_examples/episode_0/cam-5.avi",
    # "/home/selamg/beadsight/data/stonehenge_examples/episode_0/cam-6.avi"
    "/home/selamg/beadsight/data/ssd/drawer_dataset/run/episode_1/cam-1.avi",
    "/home/selamg/beadsight/data/ssd/drawer_dataset/run/episode_1/cam-2.avi",
    "/home/selamg/beadsight/data/ssd/drawer_dataset/run/episode_1/cam-3.avi",
    "/home/selamg/beadsight/data/ssd/drawer_dataset/run/episode_1/cam-4.avi",
    "/home/selamg/beadsight/data/ssd/drawer_dataset/run/episode_1/cam-5.avi",
    "/home/selamg/beadsight/data/ssd/drawer_dataset/run/episode_1/cam-6.avi"

]

cams = [1, 2, 3, 4, 5, 6]
sizes = [(1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (800, 1280)]

cameras = MultiprocessedCameras(cams, sizes)

# Create VideoCapture objects for each camera
captures = [cv2.VideoCapture(path) for path in video_paths]

frame_num = 1
# Read the frame_num frame from each camera
for _ in range(frame_num):
    original_frames = [capture.read()[1] for capture in captures]

# Stream video from each camera
while True:
    # Read the current frame from each camera
    cam_frames = cameras.get_frames()

    for i in range(len(original_frames)):
        # Average the image pairs to create the overlay
        print(cam_frames[str(i+1)].shape, original_frames[i].shape)
        overlay = np.mean([original_frames[i], cam_frames[str(i+1)]], axis=0).astype(np.uint8)

        # Display the overlay image
        cv2.imshow(f"Overlay{i}", overlay)
        cv2.waitKey(1)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release the VideoCapture objects
for capture in captures:
    capture.release()

cameras.close()

# Close all OpenCV windows
cv2.destroyAllWindows()

