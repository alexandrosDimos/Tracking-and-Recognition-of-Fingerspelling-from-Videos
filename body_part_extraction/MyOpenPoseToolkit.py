import os
import cv2
import json
import time


class OpenPoseModule:
    def __init__(self):
        self.constructor = "OpenPose Module created"
        print(self.constructor)

    def extract_number_of_people(self, json_file):
        with open(json_file, "r") as signer_file:
            signer = json.load(signer_file)
        people = signer['people']
        number_of_people = len(people)
        return number_of_people

    def extract_hand_data(self, img, json_file):
        h, w, c = img.shape
        scale_w, scale_h = self.rescale_image_shape(img)
        people_info = []
        more_than_one_people = False
        final_body_position = 0
        threshold = 30

        try:
            with open(json_file, "r") as signer_file:
                signer = json.loads(signer_file.read())
        except json.decoder.JSONDecodeError:
            print("Error decoding JSON!")

        people = signer['people']
        number_of_people = len(people)

        if number_of_people >= 2:

            more_than_one_people = True
            img = self.draw_pose_landmarks(people, img, (0, 0, 0))

            for id, person in enumerate(people):
                eye_distance = 0
                person_info = {}
                person["id"] = id
                pose_landmarks_list = person["pose_keypoints_2d"]
                left_hand_landmark_list = person["hand_left_keypoints_2d"]
                right_hand_landmark_list = person["hand_right_keypoints_2d"]
                pose_landmarks_list = self.reshape_hand_landmarks(pose_landmarks_list)
                left_hand_landmark_list = self.reshape_hand_landmarks(left_hand_landmark_list)
                right_hand_landmark_list = self.reshape_hand_landmarks(right_hand_landmark_list)
                height_list = []
                width_list = []
                right_hand_height_list = []
                left_hand_height_list = []
                for lm in pose_landmarks_list:
                    height_list.append(lm[1])
                    width_list.append(lm[0])

                # ymin, ymax = min(height_list), max(height_list)
                # print(h)
                # print(height_list)
                ymin = min(i for i in height_list if i > 0 and i < h)
                ymin = ymin * scale_h
                ymax = max(i for i in height_list if i > 0 and i < h)
                ymax = ymax * scale_h

                xmin = min(i for i in width_list if i > 0)
                xmin = xmin * scale_w
                xmax = max(i for i in width_list if i > 0)
                xmax = xmax * scale_w
                box_width, box_height = xmax - xmin, ymax - ymin
                body_position = (xmin, ymin, box_width, box_height)

                for lm in left_hand_landmark_list:
                    left_hand_height_list.append(lm[1])
                for lm in right_hand_landmark_list:
                    right_hand_height_list.append(lm[1])

                # FOR RIGHT HAND
                if right_hand_height_list.count(0) == len(right_hand_height_list):
                    min_right_hand_height = 100000
                else:
                    min_right_hand_height = min(i for i in right_hand_height_list if i > 0)
                    max_right_hand_height = max(i for i in right_hand_height_list if i > 0)
                    if max_right_hand_height > h:
                        min_right_hand_height = 100000
                # FOR LEFT HAND
                if left_hand_height_list.count(0) == len(left_hand_height_list):
                    min_left_hand_height = 100000
                else:
                    min_left_hand_height = min(i for i in left_hand_height_list if i > 0)
                    max_left_hand_height = max(i for i in left_hand_height_list if i > 0)
                    if max_left_hand_height > h:
                        min_left_hand_height = 100000

                # STORE PERSON INFO
                height = ymax - ymin
                width = xmax - xmin
                max_hand_height = min(min_left_hand_height, min_right_hand_height)
                person_info["id"] = id
                person_info["SkeletonHeight"] = (height / h) * 100
                person_info["MaxHeight"] = max_hand_height
                person_info["BodyPosition"] = body_position
                person_info["chosen"] = 0
                person_info["area"] = height * width
                person_info["right_hand"] = right_hand_landmark_list
                person_info["left_hand"] = left_hand_landmark_list
                people_info.append(person_info)

            # print(len(people_info))
            # Discard stupid little annoying people that mess with the results
            max_area = max([person["area"] for person in people_info])
            initial_discard = []

            for per in people_info:
                area = (per["area"] / max_area) * 100
                if area > threshold:
                    initial_discard.append(per)

            sorted_people = sorted(initial_discard, key=lambda d: d["MaxHeight"])
            final_skeleton_list = []
            small_skeleton_list = []
            small_skeletons = 0
            selected_skeleton_id = 0

            for person in sorted_people:
                if person["SkeletonHeight"] > 40:
                    final_skeleton_list.append(person)
                else:
                    small_skeletons = small_skeletons + 1
                    small_skeleton_list.append(person)

            selected_skeleton = {}
            if small_skeletons == len(sorted_people):
                selected_skeleton = min(small_skeleton_list, key=lambda d: d["MaxHeight"])
                if selected_skeleton["MaxHeight"] == 100000:
                    people = [people[selected_skeleton["id"]]]
                    selected_skeleton_id = selected_skeleton["id"]
                else:
                    people = [people[selected_skeleton["id"]]]
                    selected_skeleton_id = selected_skeleton["id"]

            else:
                if len(final_skeleton_list) >= 2:
                    selected_skeleton = min(final_skeleton_list, key=lambda d: d["MaxHeight"])
                    people = [people[selected_skeleton["id"]]]
                    selected_skeleton_id = selected_skeleton["id"]
                elif len(final_skeleton_list) == 0:
                    people = []
                else:
                    selected_skeleton = final_skeleton_list[0]
                    people = [people[selected_skeleton["id"]]]
                    selected_skeleton_id = selected_skeleton["id"]

            for person in people_info:
                if selected_skeleton_id == person["id"]:
                    person["chosen"] = 1
                    # print(person["area"])

            # img = self.draw_pose_keypoints(people, img, (255, 0, 0))
            # print(sorted_people)
            # plt.imshow(img)
            # plt.show()
            final_body_position = selected_skeleton["BodyPosition"]

        elif number_of_people == 1:
            pose_landmarks_list = people[0]["pose_keypoints_2d"]
            pose_landmarks_list = self.reshape_hand_landmarks(pose_landmarks_list)
            width_list = []
            height_list = []
            for lm in pose_landmarks_list:
                width_list.append(lm[0])
                height_list.append(lm[1])

            # height_list.append(lm[1])
            # width_list.append(lm[0])
            ymin = min(i for i in height_list if i > 0 and i < h)
            ymax = max(i for i in height_list if i > 0 and i < h)

            xmin = min(i for i in width_list if i > 0)
            xmax = max(i for i in width_list if i > 0)
            box_width, box_height = xmax - xmin, ymax - ymin
            body_position = xmin, ymin, box_width, box_height
            final_body_position = body_position

        persons_hands = self.extract_and_reshape_hand_data(people, img)

        if len(people) == 0:
            final_body_position = (0, 0, 0, 0)
            # print(persons_hands)
            # all_hands.append(persons_hands)

        return persons_hands, people_info, img, number_of_people, more_than_one_people, final_body_position

    def extract_feet_position(self, poseLmList, img_height):
        probably_culprit = False

        feet_points = [poseLmList[11][1], poseLmList[24][1], poseLmList[22][1], poseLmList[23][1], poseLmList[14][1],
                       poseLmList[19][1], poseLmList[20][1], poseLmList[21][1]]
        max_height_feet_point = max(feet_points)
        if max_height_feet_point != 0:
            margin = abs(max_height_feet_point - img_height) / img_height
            print(f"Feet margin: {margin}")
            if margin > 30:
                probably_culprit = True

        return probably_culprit

    def extract_and_reshape_hand_data(self, people, img):
        persons_hands = []
        for person in people:
            # persons_hands = []
            right_hand = person["hand_right_keypoints_2d"]
            # print(right_hand)
            right_hand_exists = self.check_if_hand_is_tracked(right_hand)
            if right_hand_exists:
                right_hand = self.reshape_hand_landmarks(right_hand)
                if len(right_hand) < 21:
                    print("Less than 21 points")
                    time.sleep(2)
                hand, img = self.create_hand_dictionaries(img, right_hand, "right", True)
                persons_hands.append(hand)
            else:
                persons_hands.append(None)
            left_hand = person["hand_left_keypoints_2d"]
            # print(left_hand)
            left_hand_exists = self.check_if_hand_is_tracked(left_hand)
            if left_hand_exists:
                left_hand = self.reshape_hand_landmarks(left_hand)
                if len(left_hand) < 21:
                    print("Less than 21 points")
                    time.sleep(2)
                hand, img = self.create_hand_dictionaries(img, left_hand, "left", True)
                persons_hands.append(hand)
            else:
                persons_hands.append(None)

        if len(people) == 0:
            persons_hands.append(None)
            persons_hands.append(None)

        return persons_hands

    def check_if_hand_is_tracked(self, landmarks):
        total_sum = 0
        for lm in landmarks:
            total_sum = total_sum + lm

        if total_sum == 0:
            return False
        if total_sum != 0:
            return True

    def reshape_hand_landmarks(self, landmarks):
        landmark = []
        hand_landmarks_coordinates = []
        for lm in landmarks:
            landmark.append(lm)
            if len(landmark) == 3:
                hand_landmarks_coordinates.append(landmark)
                landmark = []
        return hand_landmarks_coordinates

    def create_hand_dictionaries(self, img, landmarks, hand_type, draw):
        h, w, c = img.shape

        scale_w, scale_h = self.rescale_image_shape(img)
        show_image = False
        # print(type)
        hand = {}
        landmark_list = []
        x_list = []
        y_list = []
        below_threshold = 0
        above_threshold = 0
        for lm in landmarks:
            if lm[2] > 0.2:
                above_threshold = above_threshold + 1
            else:
                below_threshold = below_threshold + 1

        # if above_threshold >= below_threshold:
        for id, lm in enumerate(landmarks):
            cx, cy, cz = int(lm[0] * scale_w), int(lm[1] * scale_h), lm[2]
            landmark_list.append([cx, cy, cz])
            x_list.append(cx)
            y_list.append(cy)
            if cx == 0 or cy == 0:
                show_image = True
            # else:
            #    print(lm[2])
        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)
        boxW, boxH = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, boxW, boxH
        center_x, center_y = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
        hand["lmList"] = landmark_list
        hand["bbox"] = bbox
        hand["center"] = (center_x, center_y)
        hand["type"] = hand_type
        hand["signing"] = 0
        point = (int((bbox[0] + bbox[2]) / scale_w), int((bbox[1] + bbox[3]) / scale_h))
        cv2.putText(img, hand["type"], point, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return hand, img

    def rescale_image_shape(self, img):
        h, w, c = img.shape

        scale_w = 1
        scale_h = 1

        if w != 640:
            scale_w = 640 / w
        if h != 360:
            scale_h = 360 / h

        return scale_w, scale_h

    def extract_json_from_dataset(self, file_name):
        json_file_name = file_name.replace("BBox", "OpenPoseJSONFilesv2")
        split_strings = json_file_name.split('\\')
        split_strings.insert(7, 'json_files')
        final_string = '\\'.join(split_strings)
        final_string = final_string.replace(".txt", ".json")
        return final_string

    def draw_hand_landmarks(self, hands, img):
        for hand in hands:
            if hand != None:
                lmList = hand["lmList"]
                for lms in lmList:

                    if lms[2] > 0.2:
                        cv2.circle(img, (int(lms[0]), int(lms[1])), 3, (0, 0, 0), cv2.FILLED)
                    else:

                        cv2.circle(img, (int(lms[0]), int(lms[1])), 3, (0, 255, 0), cv2.FILLED)
                self.manually_draw_hand_connections_from_openpose_landmarks(lmList, img)
        return img


    def manually_draw_hand_connections_from_openpose_landmarks(self, landmark_list, img):
        color = (0, 255, 255)
        cv2.line(img, (int(landmark_list[0][0]), int(landmark_list[0][1])),
                 (int(landmark_list[1][0]), int(landmark_list[1][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[0][0]), int(landmark_list[0][1])),
                 (int(landmark_list[5][0]), int(landmark_list[5][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[0][0]), int(landmark_list[0][1])),
                 (int(landmark_list[17][0]), int(landmark_list[17][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[1][0]), int(landmark_list[1][1])),
                 (int(landmark_list[2][0]), int(landmark_list[2][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[2][0]), int(landmark_list[2][1])),
                 (int(landmark_list[3][0]), int(landmark_list[3][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[3][0]), int(landmark_list[3][1])),
                 (int(landmark_list[4][0]), int(landmark_list[4][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[5][0]), int(landmark_list[5][1])),
                 (int(landmark_list[6][0]), int(landmark_list[6][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[6][0]), int(landmark_list[6][1])),
                 (int(landmark_list[7][0]), int(landmark_list[7][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[7][0]), int(landmark_list[7][1])),
                 (int(landmark_list[8][0]), int(landmark_list[8][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[9][0]), int(landmark_list[9][1])),
                 (int(landmark_list[10][0]), int(landmark_list[10][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[10][0]), int(landmark_list[10][1])),
                 (int(landmark_list[11][0]), int(landmark_list[11][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[11][0]), int(landmark_list[11][1])),
                 (int(landmark_list[12][0]), int(landmark_list[12][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[13][0]), int(landmark_list[13][1])),
                 (int(landmark_list[14][0]), int(landmark_list[14][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[14][0]), int(landmark_list[14][1])),
                 (int(landmark_list[15][0]), int(landmark_list[15][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[15][0]), int(landmark_list[15][1])),
                 (int(landmark_list[16][0]), int(landmark_list[16][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[17][0]), int(landmark_list[17][1])),
                 (int(landmark_list[18][0]), int(landmark_list[18][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[18][0]), int(landmark_list[18][1])),
                 (int(landmark_list[19][0]), int(landmark_list[19][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[20][0]), int(landmark_list[20][1])),
                 (int(landmark_list[20][0]), int(landmark_list[20][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[0][0]), int(landmark_list[0][1])),
                 (int(landmark_list[5][0]), int(landmark_list[5][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[0][0]), int(landmark_list[0][1])),
                 (int(landmark_list[9][0]), int(landmark_list[9][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[0][0]), int(landmark_list[0][1])),
                 (int(landmark_list[13][0]), int(landmark_list[13][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[0][0]), int(landmark_list[0][1])),
                 (int(landmark_list[17][0]), int(landmark_list[17][1])), color,
                 thickness=1)

    def draw_pose_landmarks(self, people, img, color):
        for id, person in enumerate(people):
            pose_landmark_list = person["pose_keypoints_2d"]
            pose_landmark_list = self.reshape_hand_landmarks(pose_landmark_list)
            for lms in pose_landmark_list:
                cv2.circle(img, (int(lms[0]), int(lms[1])), 3, color, cv2.FILLED)
            img = self.manually_draw_pose_connections_from_openpose_landmarks(pose_landmark_list, img, color)

        return img

    def manually_draw_pose_connections_from_openpose_landmarks(self, landmark_list, img, color):

        cv2.line(img, (int(landmark_list[0][0]), int(landmark_list[0][1])),
                 (int(landmark_list[15][0]), int(landmark_list[15][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[0][0]), int(landmark_list[0][1])),
                 (int(landmark_list[16][0]), int(landmark_list[16][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[15][0]), int(landmark_list[15][1])),
                 (int(landmark_list[17][0]), int(landmark_list[17][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[16][0]), int(landmark_list[16][1])),
                 (int(landmark_list[18][0]), int(landmark_list[18][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[0][0]), int(landmark_list[0][1])),
                 (int(landmark_list[1][0]), int(landmark_list[1][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[1][0]), int(landmark_list[1][1])),
                 (int(landmark_list[2][0]), int(landmark_list[2][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[2][0]), int(landmark_list[2][1])),
                 (int(landmark_list[3][0]), int(landmark_list[3][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[3][0]), int(landmark_list[3][1])),
                 (int(landmark_list[4][0]), int(landmark_list[4][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[1][0]), int(landmark_list[1][1])),
                 (int(landmark_list[5][0]), int(landmark_list[5][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[5][0]), int(landmark_list[5][1])),
                 (int(landmark_list[6][0]), int(landmark_list[6][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[6][0]), int(landmark_list[6][1])),
                 (int(landmark_list[7][0]), int(landmark_list[7][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[1][0]), int(landmark_list[1][1])),
                 (int(landmark_list[8][0]), int(landmark_list[8][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[8][0]), int(landmark_list[8][1])),
                 (int(landmark_list[9][0]), int(landmark_list[9][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[9][0]), int(landmark_list[9][1])),
                 (int(landmark_list[10][0]), int(landmark_list[10][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[10][0]), int(landmark_list[10][1])),
                 (int(landmark_list[11][0]), int(landmark_list[11][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[11][0]), int(landmark_list[11][1])),
                 (int(landmark_list[24][0]), int(landmark_list[24][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[11][0]), int(landmark_list[11][1])),
                 (int(landmark_list[22][0]), int(landmark_list[22][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[22][0]), int(landmark_list[22][1])),
                 (int(landmark_list[23][0]), int(landmark_list[23][1])), color,
                 thickness=1)

        cv2.line(img, (int(landmark_list[8][0]), int(landmark_list[8][1])),
                 (int(landmark_list[12][0]), int(landmark_list[12][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[12][0]), int(landmark_list[12][1])),
                 (int(landmark_list[13][0]), int(landmark_list[13][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[13][0]), int(landmark_list[13][1])),
                 (int(landmark_list[14][0]), int(landmark_list[14][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[14][0]), int(landmark_list[14][1])),
                 (int(landmark_list[19][0]), int(landmark_list[19][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[14][0]), int(landmark_list[14][1])),
                 (int(landmark_list[21][0]), int(landmark_list[21][1])), color,
                 thickness=1)
        cv2.line(img, (int(landmark_list[19][0]), int(landmark_list[19][1])),
                 (int(landmark_list[20][0]), int(landmark_list[20][1])), color,
                 thickness=1)

        return img
