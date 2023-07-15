import cv2
from imutils.video import WebcamVideoStream
import imutils
import time
# img=cv2.resize(cv2.imread("./qqemo/Angry.png", cv2.IMREAD_COLOR),(200,200))
# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

stream=WebcamVideoStream(src="./video.mp4").start()
while True:
    frame=stream.read()
    print(frame)

    # if frame is not None:
    #     cv2.imshow("img",frame)
    # else:
    #     continue

