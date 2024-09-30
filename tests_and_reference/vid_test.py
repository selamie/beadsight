import cv2

cap = cv2.VideoCapture('/media/selamg/DATA/beadsight/data/full_dataset/run_5/episode_4/cam-beadsight.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('fr', frame)
        cv2.waitKey(1)
        input()