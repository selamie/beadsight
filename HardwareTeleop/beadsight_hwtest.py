from beadsight_controller.beadsight_realtime import BeadSight
# from save_data import DataRecorder, cv2_dispay
import cv2


if __name__ == '__main__':

    # test1()

    beadcam = BeadSight(36)
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