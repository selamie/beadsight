# Open the video capture
import cv2
import time
import datetime

FRAME_WIDTH = 640
FRAME_HEIGHT = 512
cap = cv2.VideoCapture(36) 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'output.mp4'
fps = 30.0
frame_size = (FRAME_WIDTH, FRAME_HEIGHT)
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

# Create a text file for storing timestamps
timestamp_file = open('timestamps.txt', 'w')

frame_number = 0

previous_time = time.time()
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    # print('frame shape:', frame.shape)
    cur_time = time.time()
    # print('dt:', cur_time - previous_time, 'should be:', 1 / fps)
    previous_time = cur_time

    if ret:
        # Add timestamp to the frame
        dt = datetime.datetime.today()  # Get timezone naive now
        seconds = dt.timestamp()

        # Write the frame number to the video frame
        cv2.putText(frame, str(frame_number), (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the frame to the output file
        out.write(frame)

        # Write the timestamp and frame number to the text file
        timestamp_file.write(str(frame_number) + ',' + str(seconds) + '\n')

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    frame_number += 1

# Release the video capture and writer objects
cap.release()
# out.release()

# Close the timestamp file
# timestamp_file.close()

# Close all OpenCV windows
cv2.destroyAllWindows()
