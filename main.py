

import cv2  # Pre-built CPU-only OpenCV packages for Python.
import numpy as np  # Powerful N-dimensional arrays, Numerical computing tools
import argparse  # library used to incorporate the parsing of command line arguments
import os

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--input', type=str, required=True,
                    help="path to input video")
parser.add_argument('--output', type=str, required=True,
                    help="path to output video")
parser.add_argument('--display', type=int, required=True,
                    help="1 for display 0 don't display")
# Parse the argument
args = parser.parse_args()

#python main.py --input video.mp4 --output video_detection_result.avi --display 1

cap = cv2.VideoCapture(args.input)  # Set the target video path
# We need to set resolutions and frame rate.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_speed = int(cap.get(5))
size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in as "args.output" we give it in argument file.
result = cv2.VideoWriter(args.output,
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         frame_speed, size)

classNames = []  # Declare an empty names list
classFile = "config.names"  # Set the config file path that contain names
with open(classFile, "rt") as f:  # here we load the names from the file to the list
    classNames = f.read().rstrip("\n").split("\n")
print(classNames)  # here we print the names from the list
configPath = "config.pbtxt"  # Set the path that contain mobilenet ssd
weightsPath = "fig.pb"  # Set the path that contain frozen inference graph {weights file}

net = cv2.dnn_DetectionModel(weightsPath, configPath)  # set the opencv config + weiths files paths
# opencv global input configuration section
net.setInputSize(320, 320)
net.setInputScale(0.5 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
thres = 0.35  # Threshold to detect object
nms_threshold = 0.35

# start looping through the video {doing actions in every Frame} 
while True:  # The condition remains True as long as the video has not ended
    success, video = cap.read()  # read the video by the opencv library
    if not success:
        break
    classIds, confs, bbox = net.detect(video, confThreshold=thres)
    # initialize the Green border that displays in the video
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    # show the Green border that displays names in the video from ids founded

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(video, (x, y), (x + w, h + y), color=(255, 0, 0), thickness=2)
        cv2.putText(video, classNames[classIds[i] - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if args.display == 0:
        result.write(video)  # write the result to a result file
    else:
        cv2.imshow("IEEE I.A", video)  # Show the form that contain the result
        result.write(video)  # write the result to a result file
        cv2.waitKey(1)
print("[INFO] detection is complete")
file_size = os.stat(str(args.output))
print("[INFO] the video is saved in " + str(args.output))
print("[INFO] Size of file :", file_size.st_size / 1048576, "Mbytes")
