from beadsight_realtime import BeadSight
# from save_data import DataRecorder, cv2_dispay
import cv2


if __name__ == '__main__':

    # test1()

    for i in range(40):
        try:
            beadcam = BeadSight(i)
            while(True):
                r, im = beadcam.get_frame()
                if r:
                    # cv2.imshow('og',og)
                    cv2.imshow('unwarped',im)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("CAM_NUM:",i)
                        break
                else:
                    break
            beadcam.cap.release()
            cv2.destroyAllWindows()
        except:
            print(i, ' is not cam')
            Exception()