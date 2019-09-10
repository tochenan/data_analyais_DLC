import cv2
import numpy as np
import os.path

# function select_roi
# return the coordinate of ROI
# features could be added to extract more than one ROI from the videoframes


videopath='C:\\Users\\caom\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3609.4d\\05_07_2019\\BRAC36094d 05_07_2019 12_37_39 1_trimmed.mp4'
frame='C:\\Users\\caom\\Desktop\\Mingran\\data processing\\score_3.png'

def select_roi(videopath):

    vidcap = cv2.VideoCapture(videopath)
    bool,frame=vidcap.read()
    count = 0

    if __name__=='__main__':
        #read the first frame of the trimmed video
        im=cv2.imread(frame)
        #select region of interest
        fromcenter=False
        showCrosshair=False
        r=cv2.selectROI('select_nest',im,fromcenter,showCrosshair)
        #crop image
        imCrop=im[int(r[1]):int(r[1]+r[3]),int(r[0])]
        #display cropped image
        cv2.imshow('nest',imCrop)
        key = cv2.waitKey(0)
        cap.realease()

        '''if key == ord('q')
            cv2.destroyAllWindows()'''

        return [int(r[0]), int(r[1])+int(r[3])]
