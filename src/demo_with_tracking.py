from argparse import ArgumentParser
import numpy as np
import cv2
import onnxruntime
import sys
from pathlib import Path
from collections import deque
import datetime
import pandas as pd
import random
from os import listdir
from os.path import isfile, join

#local imports
from face_detector import FaceDetector
from utils import draw_axis
from NodeShakeMode import NodShakeMode, NodShakeHMM


root_path = str(Path(__file__).absolute().parent.parent)

COLLECT_DATA = False  # whether to collect data or not

MAXLEN = 9 * 10  # max length of deque object used for tracking movement
# my mac currently has 9 frames per second

EULER_ANGLES = ["yaw", "pitch", "roll"]
HEAD_POSE = ["nod", "shake", "other"]

start_idx = dict()
all_files = [f for f in listdir("collected_data") if isfile(join("collected_data", f))]
for pose in HEAD_POSE:
    if MAXLEN != 9 * 4:
        pose_list = [int(f.split('_')[-1]) for f in all_files if '%s_%s_' %(pose, str(MAXLEN)) in f]
    else:
        pose_list = [int(f.split('_')[-1]) for f in all_files if pose in f]
    start_idx[pose] = max(pose_list) + 1 if pose_list else 0
print(start_idx)

def _main(cap_src):

    prev_data = []
    data = []
    mode2 = NodShakeMode(prev_data, data)
    hmm_model = NodShakeHMM(maxlen=15)

    cap = cv2.VideoCapture(cap_src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_d = FaceDetector()

    eye_model = f'{root_path}/pretrained/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

    sess = onnxruntime.InferenceSession(f'{root_path}/pretrained/fsanet-1x1-iter-688590.onnx')

    sess2 = onnxruntime.InferenceSession(f'{root_path}/pretrained/fsanet-var-iter-688590.onnx')

    tracking_dict = {k:deque(maxlen=MAXLEN) for k in ['yaw', 'pitch', 'roll']}
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    collect_head_pose = None

    prev_head_pose_hmm = 'stationary'

    print('Processing frames, press q to exit application...')
    while True:
        ret,frame = cap.read()
        if(not ret):
            print('Could not capture a valid frame from video source, check your cam/video value...')
            break
        #get face bounding boxes from frame
        if COLLECT_DATA:
            face_bb = face_d.get(frame)
            head_pose = ""
        else:
            face_bb, data = face_d.detect_face_and_eyes_enhanced(frame, cv2.CascadeClassifier(eye_model))

            # mode2.set_data(prev_data, data)
            # head_pose = mode2.apply()
            # prev_data.append(data)

        for (x1, y1, x2, y2) in face_bb:
            face_roi = frame[y1:y2+1,x1:x2+1]

            #preprocess headpose model input
            face_roi = cv2.resize(face_roi,(64,64))
            face_roi = face_roi.transpose((2,0,1))
            face_roi = np.expand_dims(face_roi,axis=0)
            face_roi = (face_roi-127.5)/128
            face_roi = face_roi.astype(np.float32)  # -> (1, 3, 64, 64)

            #get headpose
            res1 = sess.run(["output"], {"input": face_roi})[0]
            res2 = sess2.run(["output"], {"input": face_roi})[0]

            yaw, pitch, roll = np.mean(np.vstack((res1, res2)), axis=0)

            tracking_dict["yaw"].append(yaw)
            tracking_dict["pitch"].append(pitch)
            tracking_dict["roll"].append(roll)

            if not COLLECT_DATA:
                data.add_euler_angles(yaw, pitch, roll)
                mode2.set_data(prev_data, data)
                head_pose = mode2.apply()
                prev_data.append(data)
                hmm_model.add_data(data)
                new_head_pose_hmm = hmm_model.determine_pose()
                if new_head_pose_hmm == 'stationary' or prev_head_pose_hmm == new_head_pose_hmm:
                    head_pose_hmm = new_head_pose_hmm
                else:
                    head_pose_hmm = prev_head_pose_hmm
                prev_head_pose_hmm = new_head_pose_hmm
            # head_pose = ''
            print(datetime.datetime.now(), yaw, pitch, roll)
            # print(np.std(tracking_dict["yaw"]), np.std(tracking_dict["pitch"]), np.std(tracking_dict["roll"]))

            # # Nodding is when pitch is changing fairly sinusoidal while roll and yaw stays relatively consistent
            # if np.std(tracking_dict["yaw"]) < 3 and np.std(tracking_dict["roll"]) < 3 and np.std(tracking_dict["pitch"]) > 3:
            #     head_pose = 'NOD'


            draw_axis(frame, yaw, pitch, roll, tdx=(x2-x1)//2+x1, tdy=(y2-y1)//2+y1, size=50)

            # cv2.putText(frame, head_pose, (x1, y1), font, 2, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, head_pose_hmm, (x2, y2), font, 2, (0, 0, 0), 3, cv2.LINE_AA)

            #draw face bb
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            if COLLECT_DATA and len(tracking_dict["yaw"]) == MAXLEN:
                if not collect_head_pose:
                    collect_head_pose = random.choice(HEAD_POSE)
                else:
                    df = pd.DataFrame.from_dict(tracking_dict)
                    df.to_csv(
                        'collected_data/%s_%s_%s'%(
                            collect_head_pose,
                            MAXLEN,
                            start_idx[collect_head_pose]
                        ), 
                        index=False,
                    )
                    start_idx[collect_head_pose] += 1
                collect_head_pose = random.choice(HEAD_POSE)
                for angle in EULER_ANGLES:
                    tracking_dict[angle].clear()
                input("Enter to continue and do head pose: %s"%collect_head_pose)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1)&0xFF
        if(key == ord('q')):
            break




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="Path of video to process i.e. /path/to/vid.mp4")
    parser.add_argument("--cam", type=int, default=None,
                        help="Specify camera index i.e. 0,1,2...")
    args = parser.parse_args()
    cap_src = args.cam if args.cam is not None else args.video
    if(cap_src is None):
        print('Camera or video not specified as argument, selecting default camera node (0) as input...')
        cap_src = 0
    _main(cap_src)
