import numpy as np
import cv2

def video_from_behaviour_onset_frame(videopath, frame_number):
    path = 'C:\\Users\\caom\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3609.4d\\05_07_2019\\BRAC36094d 05_07_2019 12_37_39 1_trimmed.mp4'
    cap = cv2.VideoCapture(path)
    fps=30

    while(True):

        cap.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_number-1)
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)

        #waitkey()specify the amount of time in milliseconds that the initial frame wait for the consecutive frame to show up
        #to calculate int(1/fps)*1000
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
        amount_of_frames = cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
    # When everything done, release the capture
    cap.release()
    behaviour = input('please specify a behaviour e.g grooming, crouching, nest-building, chemoinvestigation, if a behaviour cannot be extracted, please type N/A')

    cv2.destroyAllWindows()
    return dict(beahviour = behaviour,
                amount_of_frames = amount_of_frames,
                latency = frame_number)
