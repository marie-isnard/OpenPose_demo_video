
import cv2 as cv
import numpy as np
import argparse
import time 
import matplotlib.pyplot as plt
import ntpath
from params import *

t = time.time()

net = cv.dnn.readNet(args.proto, args.model)

cap = cv.VideoCapture(args.input if args.input else 0)

frameWidth = int(cap.get(3))
frameHeight = int(cap.get(4))

print('frameWidth :', frameWidth, 'frameHeight :', frameHeight)

output_video = cv.VideoWriter(os.path.join(output_path, 'output_{}_{}_{}_test.avi'.format(video_name, str(inScale)[0:5], args.dataset)),cv.VideoWriter_fourcc("M", "J", "P", "G"), 20, (300,300))

while cap.isOpened():
    hasFrame, frame = cap.read()
    print('reading')
    if not hasFrame:
        print('no more frames! Program shutdown')
        cv.waitKey()
        break

    inp = cv.dnn.blobFromImage(frame, inScale, (frameWidth, frameHeight),(0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    assert(len(BODY_PARTS) <= out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            output_video.write(frame)
            print('writing')

t, _ = net.getPerfProfile()
cap.release()
output_video.release()
print("Total time taken by the net : {:.3f}".format(time.time() - t))
print('Treatment done, you can find the output video at : ', output_path)





