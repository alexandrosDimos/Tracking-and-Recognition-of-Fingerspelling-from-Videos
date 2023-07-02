import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import os
import json


class HandDetector:
    def __init__(self, mode=False, model_complexity=1, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.detectionCon = detection_con
        self.trackCon = tracking_con
        #  self.detector = MyHolisticDetector(mode=self.mode, model_complexity=model_complexity, detection_con=self.detectionCon,
        #                                   track_con=self.trackCon)
        self.mpHolistic = mp.solutions.holistic
        self.holistics = self.mpHolistic.Holistic(mode, model_complexity=model_complexity,
                                                  min_detection_confidence=detection_con,
                                                  min_tracking_confidence=tracking_con)
        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.results = None

    def find_hands(self, img, path, image_name, create_jsons, draw=True):
        hands, body_position, body_keypoints, img = self.find_holistics(img, path, image_name, create_jsons, draw)
        img = self.rescale_image(img)

        if len(hands) == 1:
            if hands[0]["type"] == "right":
                hands.append(None)
            elif hands[0]["type"] == "left":
                hands.insert(0, None)
        if len(hands) == 0:
            hands.append(None)
            hands.append(None)

        return img, hands, body_position, body_keypoints

    def rescale_image(self, img):
        img = cv2.resize(img, (640, 360))
        return img

    def find_holistics(self, img, path, image_name, create_jsons, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False  # This is to improve performance
        self.results = self.holistics.process(imgRGB)
        imgRGB.flags.writeable = True

        person_keypoints = {}

        all_hands = []
        body_position = (0, 0, 0, 0)

        if self.results.right_hand_landmarks:
            hand, right_keypoints = self.create_hand_dictionaries(imgRGB, self.results.right_hand_landmarks.landmark,
                                                                  "right", draw)
            all_hands.append(hand)
        else:
            right_keypoints = [[0, 0, 0, 0]] * 21

        if self.results.left_hand_landmarks:
            hand, left_keypoints = self.create_hand_dictionaries(imgRGB, self.results.left_hand_landmarks.landmark,
                                                                 "left", draw)
            all_hands.append(hand)
        else:
            left_keypoints = [[0, 0, 0, 0]] * 21

        if self.results.pose_landmarks:
            body_position, body_keypoints = self.extract_body_position(imgRGB, self.results.pose_landmarks.landmark)
            draw = True
            if draw:
                self.mpDraw.draw_landmarks(imgRGB, self.results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS)
        else:
            body_keypoints = [[0, 0, 0, 0]] * 33
        if create_jsons:
            person_keypoints["right_hand"] = right_keypoints
            person_keypoints["left_hand"] = left_keypoints
            person_keypoints["body_pose_keypoints"] = body_keypoints
            parent_dir = path
            directory = "json"
            # # Path
            new_path = os.path.join(parent_dir, directory)

            if not os.path.exists(new_path):
                os.makedirs(new_path)
                print("Directory ", new_path, " Created ")
            image_name = image_name.split("\\")[-1]
            image_name = image_name.split(".")[0]
            json_name = image_name + ".json"
            filename = os.path.join(new_path, json_name)

            with open(filename, 'w') as file_object:  # open the file in write mode
                json.dump(person_keypoints, file_object)

        return all_hands, body_position, body_keypoints, imgRGB

    def create_hand_dictionaries(self, img, landmarks, type, draw):
        h, w, c = img.shape

        scale_w, scale_h = self.rescale_image_dimensions(img)

        hand = {}
        landmark_list = []
        landmark_list_for_dict = []
        x_list = []
        y_list = []
        for id, lm in enumerate(landmarks):
            cx, cy, cz, cvisibility = int(lm.x * w * scale_w), int(lm.y * h * scale_h), lm.z * 100, lm.visibility
            landmark_list.append([cx, cy, cz])
            landmark_list_for_dict.append([cx, cy, cz, cvisibility])
            x_list.append(cx)
            y_list.append(cy)

        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)

        box_width, box_height = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, box_width, box_height
        center_x, center_y = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
        hand["lmList"] = landmark_list
        hand["bbox"] = bbox
        hand["center"] = (center_x, center_y)
        hand["type"] = type
        nose_lm = self.results.pose_landmarks.landmark[0]
        hand["nosePos"] = (nose_lm.x, nose_lm.y, nose_lm.z, nose_lm.visibility)
        nose_lm_x, nose_lm_y = int(nose_lm.x * w * scale_w), int(nose_lm.y * h * scale_h)
        hand_point = np.array((center_x, center_y))
        face_point = np.array((nose_lm_x, nose_lm_y))
        hand["noseDistance"] = np.sqrt(np.sum(np.square(face_point - hand_point)))
        hand["signing"] = 0

        if draw and type == "right":
            point = (int((bbox[0] - 13) / scale_w), int((bbox[1] - 13) / scale_h))
            self.mpDraw.draw_landmarks(img, self.results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS,
                                       self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                                       self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))
            cv2.putText(img, hand["type"], point, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        if draw and type == "left":
            point = (int((bbox[0] - 13) / scale_w), int((bbox[1] - 13) / scale_h))
            self.mpDraw.draw_landmarks(img, self.results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS,
                                       self.mpDraw.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1),
                                       self.mpDraw.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1))
            cv2.putText(img, hand["type"], point, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return hand, landmark_list_for_dict

    def extract_body_position(self, img, landmarks):
        h, w, c = img.shape

        scale_w, scale_h = self.rescale_image_dimensions(img)
        lm_list_for_dict = []
        x_list = []
        y_list = []
        for id, lm in enumerate(landmarks):
            cx, cy, cz, cvisibility = int(lm.x * w * scale_w), int(lm.y * h * scale_h), lm.z * 100, lm.visibility
            lm_list_for_dict.append([cx, cy, cz, cvisibility])
            if cvisibility > 0.2:
                x_list.append(cx)
                y_list.append(cy)

        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)
        box_width, box_height = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, box_width, box_height

        body_position = (xmax + xmin) / 2

        return bbox, lm_list_for_dict

    def draw_pose_lms(self, img, landmarks):
        h, w, c = img.shape
        hand = {}

        landmark_list = []
        x_list = []
        y_list = []
        for id, lm in enumerate(landmarks):
            cx, cy, cz, vis = int(lm.x * w), int(lm.y * h), lm.z, lm.visibility

            landmark_list.append([cx, cy, cz])
            x_list.append(cx)
            y_list.append(cy)
            cv2.circle(img, (cx, cy), 3, (255, 165, 0), cv2.FILLED)

    def rescale_image_dimensions(self, img):
        h, w, c = img.shape

        scale_w = 1
        scale_h = 1
        # img = cv2.resize(img, (640, 360))
        if w != 640:
            scale_w = 640 / w
        if h != 360:
            scale_h = 360 / h

        return scale_w, scale_h
