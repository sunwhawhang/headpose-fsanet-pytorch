# Standard library imports
from pathlib import Path
import glob
import time
import numpy as np
import cv2
import sys

#top_level_dir path
root_path = Path(__file__).parent.parent

conf_value = 0.7
face_left_color = (138, 0, 138)
face_right_color = (138, 138, 138)
face_middle_color = (0, 138, 0)
face_eps = 10
middle_point = []

class FaceDetector:
    """
    This class is used for detecting face.
    """

    def __init__(self):

        """
        Constructor of class
        """

        config_path = root_path.joinpath("pretrained/",
                                "resnet10_ssd.prototxt")
        face_model_path = root_path.joinpath("pretrained/",
                "res10_300x300_ssd_iter_140000.caffemodel")

        self.detector = cv2.dnn.readNetFromCaffe(str(config_path),
                                            str(face_model_path))

        #detector prediction threshold
        self.confidence = 0.7


    def get(self,img):
        """
        Given a image, detect faces and compute their bb

        """
        bb =  self._detect_face_ResNet10_SSD(img)

        return bb

    def _detect_face_ResNet10_SSD(self,img):
        """
        Given a img, detect faces in it using resnet10_ssd detector

        """

        detector = self.detector
        (h, w) = img.shape[:2]
        # construct a blob from the image
        img_blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(img_blob)
        detections = detector.forward()

        (start_x, start_y, end_x, end_y) = (0,0,0,0)
        faces_bb = []
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            for i in range(0, detections.shape[2]):

                score = detections[0, 0, i, 2]

                # ensure that the detection greater than our threshold is
                # selected
                if score > self.confidence:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    box = box.astype("int")
                    (start_x, start_y, end_x, end_y) = box

                    # extract the face ROI and grab the ROI dimensions
                    face = img[start_y:end_y, start_x:end_x]

                    (fh, fw) = face.shape[:2]
                    # ensure the face width and height are sufficiently large
                    if fw < 20 or fh < 20:
                        pass
                    else:
                        faces_bb.append(box)

        if(len(faces_bb)>0):
            faces_bb = np.array(faces_bb)

        return faces_bb

    def detect_face_and_eyes_enhanced(self, img, eye_classifier):
        """
        An enhanced method for face and eye detection. Faces are detected through a pretrained network. Every eye is
        searched for separately in a vertical face half. The reference point is computed from both eye positions if
        available, or estimated if no two eyes are detected.

        :param net: the network used for face detection
        :param eye_classifier: the classifier used for eye detection
        :return: the data of the processed image frame
        """
        net = self.detector

        left_ey = 0
        right_ey = 0
        left_ew = 0
        right_ew = 0
        left_eh = 0
        right_eh = 0

        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        left_faces = []
        right_faces = []

        faces_bb = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < conf_value:
                print("INFO: Face found with confidence below threshold. Confidence value: ", confidence)

            if confidence >= conf_value:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                box = box.astype("int")
                (startX, startY, endX, endY) = box
                faces_bb.append(box)
                text = "Face: {:.2f}%".format(confidence * 100)
                y = startY - face_eps if startY - face_eps > 10 else startY + face_eps

                # left face half
                # cv2.rectangle(img, (startX, startY), (endX - int((endX - startX) / 2) + face_eps, endY),
                #               face_left_color, 4)

                # right face half
                # cv2.rectangle(img, (startX + int((endX - startX) / 2) - face_eps, startY), (endX, endY),
                #               face_right_color, 4)

                faces.append([startX, startY, endX - startX, endY - startY])
                left_faces.append(
                    [startX, startY, (endX - int((endX - startX) / 2) + face_eps - startX), endY - startY])
                right_faces.append([startX + int((endX - startX) / 2) - face_eps, startY, endX - startX, endY - startY])
                # cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, face_left_color, 2)
                break

        # for left eye
        left_ex = sys.maxsize
        x_left_face = 0
        y_left_face = 0
        for (x, y, w, h) in left_faces:

            # detect eyes
            roi = img[y:y + h, x:x + w]
            eyes = eye_classifier.detectMultiScale(roi)
            i = 0
            for (ex, ey, ew, eh) in eyes:
                # get most left eye by x-coordinate
                if ex < left_ex:
                    left_ex = ex
                    left_ey = ey
                    left_ew = ew
                    left_eh = eh
                    x_left_face = x
                    y_left_face = y
                i = i + 1
            # if left_ex < sys.maxsize:
            #     cv2.rectangle(roi, (left_ex, left_ey), (left_ex + left_ew, left_ey + left_eh), face_left_color, 4)

        # for right eye
        right_ex = 0
        x_right_face = 0
        y_right_face = 0
        for (x, y, w, h) in right_faces:
            # detect eyes
            roi = img[y:y + h, x:x + w]
            eyes = eye_classifier.detectMultiScale(roi)
            i = 0
            for (ex, ey, ew, eh) in eyes:
                # get most right eye by x-coordinate
                if ex > right_ex:
                    right_ex = ex
                    right_ey = ey
                    right_ew = ew
                    right_eh = eh
                    x_right_face = x
                    y_right_face = y
                i = i + 1
            # if right_ex > 0:
            #     cv2.rectangle(roi, (right_ex, right_ey), (right_ex + right_ew, right_ey + right_eh), face_right_color,
            #                   4)

        global middle_point
        if left_ex < sys.maxsize and right_ex > 0:

            # point for left eye
            left_point = (x_left_face + left_ex, y_left_face + left_ey + int(round(left_eh / 2)))
            cv2.circle(img, left_point, 10, face_left_color, -1)

            # point for right eye
            right_point = (x_right_face + right_ex + right_ew, y_right_face + right_ey + int(round(right_eh / 2)))
            cv2.circle(img, right_point, 10, face_right_color, -1)

            middle_point = int(round((left_point[0] + right_point[0]) / 2)), int(round((left_point[1] + right_point[1])
                                                                                       / 2))
        else:
            if len(faces) > 0:
                for (x, y, w, h) in faces:  # relevant only if a face but no eyes are recognized
                    middle_point = int(round((x + (x + w)) / 2)), int(round((y + (y + h)) / 2)) - 2 * face_eps
            if not len(middle_point) > 0:  # no middle point retrieved before, if no face and no eyes are found
                middle_point = (0, 0)
        # cv2.circle(img, middle_point, 10, face_middle_color, -1)

        return faces_bb, ProcessedImage(img, middle_point[0], middle_point[1], faces_bb)


class ProcessedImage:
    """
    The data of the processed image frame, which is calculated from face and eye positions.
    """

    def __init__(self, frame, x_middle, y_middle, faces_bb):
        """
        The constructor that sets the initialization parameters for the processed image class.

        :param frame: the processed frame after face and eye detection
        :param x_middle: the x coordinate of the retrieved reference point
        :param y_middle: the y coordinate of the retrieved reference point
        """
        self.frame = frame
        self.x_middle = x_middle
        self.y_middle = y_middle
        
        (x1, y1, x2, y2) = faces_bb[0] if faces_bb else (0, 0, 0, 0)
        self.x_middle_relative = (x_middle - x1) / (x2 - x1)
        self.y_middle_relative = (y_middle - y1) / (y2 - y1)

        self.use_euler_angles = False

    def add_euler_angles(self, yaw, pitch, roll):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.use_euler_angles = True