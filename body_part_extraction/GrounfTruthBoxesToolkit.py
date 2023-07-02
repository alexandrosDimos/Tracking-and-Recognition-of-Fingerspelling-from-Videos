import cv2
import numpy as np
#from handSequenceEditor import handSequenceEditor


class HandAndBoundingBoxesTools:
    def __init__(self):
        self.constructor = "This module contains methods used across all the programs"
        #print(self.constructor)

    def check_hand_descrepancies(self, hands, ground_truth_boxes):
        images_with_more_hands_detected = 0
        images_with_less_hands_detected = 0
        images_with_equal_hands_detected = 0
        images_with_zero_hands_detected = 0

        if len(hands) == len(ground_truth_boxes):
            images_with_equal_hands_detected = images_with_equal_hands_detected + 1
        elif len(hands) > len(ground_truth_boxes):
            # print("I have 3 hands")
            images_with_more_hands_detected = images_with_more_hands_detected + 1
        elif len(ground_truth_boxes) > len(hands) and len(hands) != 0:
            images_with_less_hands_detected = images_with_less_hands_detected + 1
            # print("failed_to detect hands")
        elif len(hands) == 0:
            images_with_zero_hands_detected = 1
            #hand_descrepancy = True

        return images_with_more_hands_detected, images_with_less_hands_detected, images_with_equal_hands_detected, images_with_zero_hands_detected

    def calculate_IOU(self, bbox1, bbox2):

        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        hand_rect_1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        hand_rect_2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        iou = interArea / float(hand_rect_1_area + hand_rect_2_area - interArea)

        return iou

    def extract_img_from_dataset(self, file_name):
        if file_name.endswith(".txt"):
            image_file_name = file_name.replace(".txt", ".jpg")
            image_file_name = image_file_name.replace("BBox", "ChicagoFSWild-Frames")
            img = cv2.imread(image_file_name)
        else:
            img = cv2.imread(file_name)

        return img

    def extract_info_from_bbox_file(self, root, file):
        #file_name = root + '\\' + file
        bbox_file = open(file, 'r')
        lines = bbox_file.readlines()
        hand_bboxes = self.string_to_nums(lines)
        signing_hands = self.number_of_signing_hands(hand_bboxes)

        return  hand_bboxes, signing_hands

    def draw_bounding_boxes(self, img, ground_truth_bounding_boxes, hands):
        for hand in hands:
            if hand is not None:
                bbox = hand["bbox"]  # bbox has info about x,y,w,h
                # hand_rect = (int(scale_w * (bbox[0] - 15)), int(scale_h * (bbox[1] - 15)), int(scale_w * (bbox[0] + bbox[2] + 15)),int(scale_h * (bbox[1] + bbox[3] + 15)))
                hand_rect = ((bbox[0] - 13), (bbox[1] - 13), (bbox[0] + bbox[2] + 13), (bbox[1] + bbox[3] + 13))
                cv2.rectangle(img, (hand_rect[0], hand_rect[1]), (hand_rect[2], hand_rect[3]), (0, 128, 0), 2)

        if ground_truth_bounding_boxes is not None:
            for bbox in ground_truth_bounding_boxes:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (128, 0, 0), 2)
                cv2.putText(img, str(bbox[4]), (bbox[0], bbox[1] + 25), cv2.FONT_HERSHEY_PLAIN, 2, (128, 0, 0), 2)

        return img

    def number_of_signing_hands(self,hand_bboxes):
        signing_hands = 0
        for bbox in hand_bboxes:
            if bbox[4] == 1:
                signing_hands = signing_hands + 1
        return signing_hands

    def string_to_nums(self,line_list):
        ints = []
        hand_bboxes = []
        for line in line_list:
            numbers = line.split(',')
            for number in numbers:
                ints.append(int(number))
            hand_bboxes.append((ints[0], ints[1], ints[2], ints[3], ints[4]))
            ints = []

        return hand_bboxes


