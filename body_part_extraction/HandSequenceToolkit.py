import cv2
import numpy as np
import matplotlib.pyplot as plt
from GrounfTruthBoxesToolkit import HandAndBoundingBoxesTools
from PIL import Image

metrics_module = HandAndBoundingBoxesTools()


class HandSequenceToolkit:
    def __init__(self, img_seq, hand_sequence, ground_truth_box_sequence):
        self.img_seq = img_seq
        self.hand_sequence = hand_sequence
        self.ground_truth_box_sequence = ground_truth_box_sequence

    def extract_right_and_left_hand_sequence(self, hand_pairs):
        right_hand_seq = []
        left_hand_seq = []
        for hand in hand_pairs:
            right_hand_seq.append(hand[0])
            left_hand_seq.append(hand[1])

        return right_hand_seq, left_hand_seq

    def update_sequences(self, iou_mode=False):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        right_hand_sequence, left_hand_sequence = self.extract_right_and_left_hand_sequence(self.hand_sequence)
        updated_right_hand_sequence, show_image_right = self.check_and_calculate_undetected_hands(right_hand_sequence)
        updated_left_hand_sequence, show_image_left = self.check_and_calculate_undetected_hands(left_hand_sequence)

        # self.discard_sequences_with_low_detection_frame_ratio(updated_right_hand_sequence)
        # self.discard_sequences_with_low_detection_frame_ratio(updated_left_hand_sequence)

        idx = 0
        landmark_list = []
        draw_lms = False

        if draw_lms:
            for right_hand, left_hand in zip(updated_right_hand_sequence, updated_left_hand_sequence):
                if right_hand is not None:
                    landmark_list_right = right_hand["lmList"]
                    landmark_list = landmark_list + landmark_list_right
                if left_hand is not None:
                    landmark_list_left = left_hand["lmList"]
                    landmark_list = landmark_list + landmark_list_left

                for lms in landmark_list:
                    cv2.circle(self.img_seq[idx], (lms[0], lms[1]), 3, (0, 0, 0), cv2.FILLED)
                idx = idx + 1
                landmark_list = []

        hand_info, updated_right_hand_sequence, updated_left_hand_sequence = self.determine_signing_hand(
            updated_right_hand_sequence, updated_left_hand_sequence)
        updated_sequence = [list(hands) for hands in zip(updated_right_hand_sequence, updated_left_hand_sequence)]

        iou_right = 0
        iou_left = 0
        if iou_mode:
            if hand_info[0] == 1:
                iou_right, true_positives, false_positives, false_negatives = self.extract_metrics_to_dermine_algorithm_efficiency(
                    updated_right_hand_sequence, self.ground_truth_box_sequence, self.img_seq)
                print(f"Right is signing and has an IOU of {iou_right}")
            if hand_info[1] == 1:
                iou_left, true_positives, false_positives, false_negatives = self.extract_metrics_to_dermine_algorithm_efficiency(
                    updated_left_hand_sequence, self.ground_truth_box_sequence, self.img_seq)
                print(f"Left is signing and has an IOU of {iou_left}")

        total_iou = iou_right + iou_left
        signing_hand_sequence = [None] * len(updated_right_hand_sequence)
        if hand_info[0] == 1:
            signing_hand_sequence = updated_right_hand_sequence
        if hand_info[1] == 1:
            signing_hand_sequence = updated_left_hand_sequence

        # self.plot_3d_hands(updated_right_hand_sequence,self.img_seq)

        show_sequence = 'n'
        if show_sequence == 'y':
            for id, img in enumerate(self.img_seq):
                plt.title(id + 1)
                plt.imshow(img)
                plt.show()
                save = 'y'
                if save == 'y':
                    cv2.imwrite("frame_sequence/" + str(id + 1) + ".jpeg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    # For Mediapipe set cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return updated_sequence, signing_hand_sequence, hand_info, total_iou, true_positives, false_positives, false_negatives

    def check_and_calculate_undetected_hands(self, sequence):
        show_image = False
        for idx, hand in enumerate(sequence):
            if idx <= len(sequence) - 3:
                window = [hand, sequence[idx + 1], sequence[idx + 2]]
                if window.count(None) > 1 or window[0] == None or window[2] == None:
                    continue
                elif window.count(None) == 1:
                    show_image = True
                    #  print("I found a missing {} hand in frame {}".format(hand["type"],idx+1))
                    hand = {}
                    first_hand_landmark_list = window[0]["lmList"]
                    third_hand_landmark_list = window[2]["lmList"]
                    new_landmark_list = []
                    x_list = []
                    y_list = []
                    for id in range(len(first_hand_landmark_list)):
                        lm1 = first_hand_landmark_list[id]
                        lm2 = third_hand_landmark_list[id]
                        new_landmarks = [int((lm1[0] + lm2[0]) / 2), int((lm1[1] + lm2[1]) / 2),
                                         int((lm1[2] + lm2[2]) / 2)]
                        x_list.append(int((lm1[0] + lm2[0]) / 2))
                        y_list.append(int((lm1[1] + lm2[1]) / 2))
                        new_landmark_list.append(new_landmarks)

                    xmin, xmax = min(x_list), max(x_list)
                    ymin, ymax = min(y_list), max(y_list)
                    box_width, box_height = xmax - xmin, ymax - ymin
                    bbox = xmin, ymin, box_width, box_height
                    center_x, center_y = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
                    hand["lmList"] = new_landmark_list
                    hand["bbox"] = bbox
                    hand["center"] = (center_x, center_y)
                    hand["type"] = window[0]["type"]
                    hand["signing"] = window[0]["signing"]
                    sequence[idx + 1] = hand
                    # metricsModule.draw_bounding_boxes(self.img_seq[idx+1], [], [hand])
                    ##plt.imshow(self.img_seq[idx+1])
                    # plt.show()
                    # self.draw_hand_connections(hand["lmList"],self.img_seq[idx+1],(75,0,130))

        return sequence, show_image

    def draw_hand_connections(self, lmList, img, color):
        cv2.line(img, (lmList[0][0], lmList[0][1]), (lmList[1][0], lmList[1][1]), color, thickness=1)
        cv2.line(img, (lmList[0][0], lmList[0][1]), (lmList[5][0], lmList[5][1]), color, thickness=1)
        cv2.line(img, (lmList[0][0], lmList[0][1]), (lmList[17][0], lmList[17][1]), color, thickness=1)

        cv2.line(img, (lmList[1][0], lmList[1][1]), (lmList[2][0], lmList[2][1]), color, thickness=1)
        cv2.line(img, (lmList[2][0], lmList[2][1]), (lmList[3][0], lmList[3][1]), color, thickness=1)
        cv2.line(img, (lmList[3][0], lmList[3][1]), (lmList[4][0], lmList[4][1]), color, thickness=1)

        cv2.line(img, (lmList[5][0], lmList[5][1]), (lmList[6][0], lmList[6][1]), color, thickness=1)
        cv2.line(img, (lmList[6][0], lmList[6][1]), (lmList[7][0], lmList[7][1]), color, thickness=1)
        cv2.line(img, (lmList[7][0], lmList[7][1]), (lmList[8][0], lmList[8][1]), color, thickness=1)

        cv2.line(img, (lmList[9][0], lmList[9][1]), (lmList[10][0], lmList[10][1]), color, thickness=1)
        cv2.line(img, (lmList[10][0], lmList[10][1]), (lmList[11][0], lmList[11][1]), color, thickness=1)
        cv2.line(img, (lmList[11][0], lmList[11][1]), (lmList[12][0], lmList[12][1]), color, thickness=1)

        cv2.line(img, (lmList[13][0], lmList[13][1]), (lmList[14][0], lmList[14][1]), color, thickness=1)
        cv2.line(img, (lmList[14][0], lmList[14][1]), (lmList[15][0], lmList[15][1]), color, thickness=1)
        cv2.line(img, (lmList[15][0], lmList[15][1]), (lmList[16][0], lmList[16][1]), color, thickness=1)

        cv2.line(img, (lmList[17][0], lmList[17][1]), (lmList[18][0], lmList[18][1]), color, thickness=1)
        cv2.line(img, (lmList[18][0], lmList[18][1]), (lmList[19][0], lmList[19][1]), color, thickness=1)
        cv2.line(img, (lmList[20][0], lmList[20][1]), (lmList[20][0], lmList[20][1]), color, thickness=1)

        cv2.line(img, (lmList[5][0], lmList[5][1]), (lmList[9][0], lmList[9][1]), color, thickness=1)
        cv2.line(img, (lmList[9][0], lmList[9][1]), (lmList[13][0], lmList[13][1]), color, thickness=1)
        cv2.line(img, (lmList[13][0], lmList[13][1]), (lmList[17][0], lmList[17][1]), color, thickness=1)

    def discard_sequences_with_low_detection_frame_ratio(self, sequence):
        total_hands_in_sequence = 0
        sequence_length = len(sequence)

        for idx, hand in enumerate(sequence):
            if hand is not None:
                total_hands_in_sequence = total_hands_in_sequence + 1

        detection_frame_ratio = 100 * (total_hands_in_sequence / sequence_length)
        # print(detection_frame_ratio)
        if detection_frame_ratio < 20:
            updated_sequence = [None] * sequence_length
        else:
            updated_sequence = sequence

        return updated_sequence

    def determine_signing_hand(self, right_hand_sequence, left_hand_sequence):

        right_hand_deviation, right_slope, hand_type_right = self.calculate_total_finger_movement_in_sequence(
            right_hand_sequence, False)
        left_hand_deviation, left_slope, hand_type_left = self.calculate_total_finger_movement_in_sequence(left_hand_sequence,
                                                                                                           False)
        right_hand_height = self.calculate_height_of_hand_in_frame(right_hand_sequence)
        left_hand_height = self.calculate_height_of_hand_in_frame(left_hand_sequence)

        if hand_type_right is not None and hand_type_left is not None:
            hand = self.select_the_signing_hand((right_hand_deviation, left_hand_deviation),
                                                (right_hand_height, left_hand_height))
            right_predicted_signing_value = hand["Right"]
            left_predicted_signing_value = hand["Left"]
        else:
            if hand_type_right is None and hand_type_left is not None:
                right_predicted_signing_value = 0
                left_predicted_signing_value = 1
            elif hand_type_left is None and hand_type_right is not None:
                right_predicted_signing_value = 1
                left_predicted_signing_value = 0
            else:
                right_predicted_signing_value = 0
                left_predicted_signing_value = 0

        for hand in right_hand_sequence:
            if hand is not None:
                hand["signing"] = right_predicted_signing_value
        for hand in left_hand_sequence:
            if hand is not None:
                hand["signing"] = left_predicted_signing_value

        hand_info = [right_predicted_signing_value, left_predicted_signing_value]
        return hand_info, right_hand_sequence, left_hand_sequence

    @staticmethod
    def calculate_distance_from_face(sequence):
        total_nose_distance = 0
        total_hands_in_sequence = 0
        average_nose_distance = 0
        hand_type = None
        for idx, hand in enumerate(sequence):
            if hand is not None:
                total_nose_distance = total_nose_distance + hand["noseDistance"]
                hand_type = hand["type"]
                total_hands_in_sequence = total_hands_in_sequence + 1

        if total_hands_in_sequence != 0:
            average_nose_distance = total_nose_distance / total_hands_in_sequence

        return average_nose_distance, hand_type

    def calculate_total_finger_movement_in_sequence(self, sequence, draw=False):
        all_distances = []
        thumb_index_distances = []
        index_middle_distances = []
        middle_ring_distances = []
        ring_pinky_distances = []
        total_hands = []
        distance_points = []
        hand_type = None
        hands = 0

        for hand in sequence:
            if hand is not None:
                hand_type = hand["type"]
                thumb = [(hand["lmList"][4][0], hand["lmList"][4][1])]
                index = [(hand["lmList"][8][0], hand["lmList"][8][1])]
                middle = [(hand["lmList"][12][0], hand["lmList"][12][1])]
                ring = [(hand["lmList"][16][0], hand["lmList"][16][1])]
                pinky = [(hand["lmList"][20][0], hand["lmList"][20][1])]
                thumb_index_dist = self.calculate_distance_between_finger_points(thumb, index)
                index_middle_dist = self.calculate_distance_between_finger_points(index, middle)
                middle_ring_dist = self.calculate_distance_between_finger_points(middle, ring)
                ring_pinky_dist = self.calculate_distance_between_finger_points(ring, pinky)
                thumb_index_distances.append(thumb_index_dist)
                index_middle_distances.append(index_middle_dist)
                middle_ring_distances.append(middle_ring_dist)
                ring_pinky_distances.append(ring_pinky_dist)
                distance_points.append((thumb_index_dist, index_middle_dist, middle_ring_dist, ring_pinky_dist))
                total_hands.append(hands)
                distance = thumb_index_dist + index_middle_dist + middle_ring_dist + ring_pinky_dist
                all_distances.append(distance)
                hands = hands + 1
            # print(distance)

        finger_distances = [thumb_index_distances, index_middle_distances, middle_ring_distances, ring_pinky_distances]
        total_deviation = 0
        total_slope = 0
        if hands != 0:
            array_x = np.array([x for x in range(0, hands)])
            for counter, finger_dist in enumerate(finger_distances):
                if counter == 0:
                    color = 'red'
                if counter == 1:
                    color = 'green'
                if counter == 2:
                    color = 'black'
                if counter == 3:
                    color = 'blue'
                if hands == 1:
                    true_values = np.full((1, len(finger_dist)), finger_dist[0])
                    mean_square_error = np.square(np.subtract(true_values, finger_dist)).mean()
                else:
                    array_y = np.array(finger_dist)
                    fit = np.polyfit(array_x, array_y, 1)
                    true_values = fit[0] * array_x + fit[1]
                    if draw:
                        plt.plot(array_x, fit[0] * array_x + fit[1], c=color)
                    mean_square_error = np.square(np.subtract(true_values, array_y)).mean()

                    total_slope = total_slope + abs(fit[0])
                total_deviation = total_deviation + mean_square_error

        if draw:
            for idx, xe in enumerate(total_hands):
                plt.scatter(xe, thumb_index_distances[idx], c='red')
                plt.scatter(xe, index_middle_distances[idx], c='green')
                plt.scatter(xe, middle_ring_distances[idx], c='black')
                plt.scatter(xe, ring_pinky_distances[idx], c='blue')

            plt.xlabel('Frames')
            plt.ylabel('Distances')
            plt.show()

        return total_deviation, total_slope, hand_type

    @staticmethod
    def calculate_distance_between_finger_points(finger1, finger2):
        total_distance = 0
        for idx, point in enumerate(finger1):
            point2 = finger2[idx]
            distance = np.sqrt(np.sum(np.square(np.array(point) - np.array(point2))))
            total_distance = total_distance + distance

        return total_distance

    @staticmethod
    def calculate_height_of_hand_in_frame(sequence):
        average_height = 0
        total_height = 0
        hands = 0
        for idx, hand in enumerate(sequence):
            if hand is not None:
                total_height = total_height + hand["center"][1]
                hands = hands + 1
        if hands != 0:
            average_height = total_height / hands

        return average_height

    @staticmethod
    def number_of_hands_in_sequences(sequence):
        total_hands = 0
        for hand in sequence:
            if hand is not None:
                total_hands = total_hands + 1

        return (total_hands / len(sequence)) * 100

    @staticmethod
    def select_the_signing_hand(hand_deviations, hand_heights):

        hand = {
            "Right": 0,
            "Left": 0
        }
        if hand_deviations[0] == 0:
            hand["Right"] = 2
            hand["Left"] = 1
        elif hand_deviations[1] == 0:
            hand["Right"] = 1
            hand["Left"] = 2
        else:
            if max(hand_deviations) / min(hand_deviations) < 2.5:
                height_diff = abs(hand_heights[0] - hand_heights[1])
                if height_diff < 50:
                    if hand_deviations[0] > hand_deviations[1]:
                        # print("Probably right has signing value 1")
                        hand["Right"] = 1
                        hand["Left"] = 2
                    else:
                        # print("Probably left has signing value 1")
                        hand["Right"] = 2
                        hand["Left"] = 1
                else:
                    if hand_heights[0] < hand_heights[1]:
                        hand["Right"] = 1
                        hand["Left"] = 2
                    else:
                        hand["Right"] = 2
                        hand["Left"] = 1
            else:
                if hand_deviations[0] > hand_deviations[1]:
                    hand["Right"] = 1
                    hand["Left"] = 2
                else:
                    hand["Right"] = 2
                    hand["Left"] = 1

        return hand

    @staticmethod
    def extract_metrics_to_dermine_algorithm_efficiency(main_sequence, ground_truth_boxes_sequence, image_seq):

        final_iou = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for idx, hand in enumerate(main_sequence):

            if hand is not None:
                # cv2.circle(self.img_seq[idx], hand["center"], 3, (255, 255, 255), cv2.FILLED)
                bbox = hand["bbox"]
                hand_rect = ((bbox[0] - 13), (bbox[1] - 13), (bbox[0] + bbox[2] + 13), (bbox[1] + bbox[3] + 13))
                gtb_bboxes = ground_truth_boxes_sequence[idx]
                gtb_bbox_of_signing_hand = []
                for gtb_bbox in gtb_bboxes:
                    if gtb_bbox[4] == 1:
                        gtb_bbox_of_signing_hand.append(gtb_bbox)

                if len(gtb_bbox_of_signing_hand) == 0:
                    print("No signing hand here")
                if len(gtb_bbox_of_signing_hand) == 1:
                    iou = metrics_module.calculate_IOU(hand_rect, gtb_bbox_of_signing_hand[0])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (hand_rect[2], hand_rect[1])
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    # cv2.putText(image_seq[idx], str("{:.2f}".format(iou)), org, font, fontScale, color, thickness,cv2.LINE_AA)
                    if iou > 0.5:
                        true_positives = true_positives + 1
                        # print("True positive")
                    else:
                        false_positives = false_positives + 1
                    # plt.imshow(image_seq[idx])
                    # plt.show()
                    # print(f"False positive{iou}")
                    final_iou = final_iou + iou
                if len(gtb_bbox_of_signing_hand) == 2:
                    temp_list = []
                    hand_centroid = hand["center"]
                    ################Probably not gonna need this code#################
                    for id, bbox in enumerate(gtb_bbox_of_signing_hand):
                        bounding_box_centroid = np.array((int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)))
                        distance = np.sqrt(np.sum(np.square(bounding_box_centroid - np.array(hand_centroid))))
                        temp_list.append((id, distance))
                    min_distance_tuple = min(temp_list, key=lambda t: t[1])
                    ###################################################################
                    if min_distance_tuple[0] == 0:
                        iou = metrics_module.calculate_IOU(hand_rect, gtb_bbox_of_signing_hand[0])
                        final_iou = final_iou + iou
                    if min_distance_tuple[0] == 1:
                        iou = metrics_module.calculate_IOU(hand_rect, gtb_bbox_of_signing_hand[1])
                        final_iou = final_iou + iou
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (hand_rect[2], hand_rect[1])
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    # cv2.putText(image_seq[idx], str("{:.2f}".format(iou)), org, font, fontScale, color, thickness,cv2.LINE_AA)
                    if iou > 0.5:
                        true_positives = true_positives + 1
                        # print("True positive")
                    else:
                        false_positives = false_positives + 1
                        # print("False positive")
            else:
                false_negatives = false_negatives + 1

        # print(final_iou)
        return final_iou, true_positives, false_positives, false_negatives

    def plot_3d_hands(self, sequence, image_seq):

        temp_sequence = self.extract_hand_coordinates(sequence, 0)
        print(temp_sequence)
        for idx1, coordinate in enumerate(temp_sequence):
            temp_sequence[idx1] = self.coordinate_regularization(temp_sequence[idx1])

        max_ele = max([max(index) for index in zip(*temp_sequence)])
        min_ele = min([min(index) for index in zip(*temp_sequence)])
        sc = self.distance_normalization(temp_sequence)
        print(max_ele)
        print(min_ele)

        for idx, hand in enumerate(sequence):
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            img = image_seq[idx]
            # plt.imshow(img)
            # plt.show()
            wrist_x = hand["lmList"][0][0]
            wrist_y = hand["lmList"][0][1]
            wrist_z = hand["lmList"][0][2]
            if hand != None:
                lmList = hand["lmList"]
                for i in range(len(lmList)):
                    lmList[i][0] = lmList[i][0] - wrist_x
                    lmList[i][1] = lmList[i][1] - wrist_y
                    lmList[i][2] = lmList[i][2] - wrist_z

                # "Min-Max Scaling"
                # for i in range(len(lmList)):
                #     lmList[i][0] = (lmList[i][0] - min_ele)/(max_ele - min_ele)
                #     lmList[i][1] = (lmList[i][1] - min_ele)/(max_ele - min_ele)
                #     lmList[i][2] = (lmList[i][2] - min_ele)/(max_ele - min_ele)

                "Hand Distance"
                for i in range(len(lmList)):
                    lmList[i][0] = lmList[i][0] / sc
                    lmList[i][1] = lmList[i][1] / sc
                    lmList[i][2] = lmList[i][2] / sc

                for lms in lmList:
                    x = lms[0]
                    y = lms[1]
                    z = lms[2]
                    # print(x,y,z)
                    ax.scatter(x, y, z)
                # ax.text(lmList[0][0], lmList[0][1], lmList[0][2], '0 0 0', size=10, zorder=2,color='k')

                # Wrist-Thumb
                ax.plot([lmList[0][0], lmList[1][0]], [lmList[0][1], lmList[1][1]], [lmList[0][2], lmList[1][2]],
                        c='black')
                ax.plot([lmList[1][0], lmList[2][0]], [lmList[1][1], lmList[2][1]], [lmList[1][2], lmList[2][2]],
                        c='black')
                ax.plot([lmList[2][0], lmList[3][0]], [lmList[2][1], lmList[3][1]], [lmList[2][2], lmList[3][2]],
                        c='black')
                ax.plot([lmList[3][0], lmList[4][0]], [lmList[3][1], lmList[4][1]], [lmList[3][2], lmList[4][2]],
                        c='black')
                # Wrist-Index
                ax.plot([lmList[0][0], lmList[5][0]], [lmList[0][1], lmList[5][1]], [lmList[0][2], lmList[5][2]],
                        c='red')
                ax.plot([lmList[5][0], lmList[6][0]], [lmList[5][1], lmList[6][1]], [lmList[5][2], lmList[6][2]],
                        c='purple')
                ax.plot([lmList[6][0], lmList[7][0]], [lmList[6][1], lmList[7][1]], [lmList[6][2], lmList[7][2]],
                        c='purple')
                ax.plot([lmList[7][0], lmList[8][0]], [lmList[7][1], lmList[8][1]], [lmList[7][2], lmList[8][2]],
                        c='purple')
                # Wrist-Middle
                ax.plot([lmList[0][0], lmList[9][0]], [lmList[0][1], lmList[9][1]], [lmList[0][2], lmList[9][2]],
                        c='red')
                ax.plot([lmList[9][0], lmList[10][0]], [lmList[9][1], lmList[10][1]], [lmList[9][2], lmList[10][2]],
                        c='green')
                ax.plot([lmList[10][0], lmList[11][0]], [lmList[10][1], lmList[11][1]], [lmList[10][2], lmList[11][2]],
                        c='green')
                ax.plot([lmList[11][0], lmList[12][0]], [lmList[11][1], lmList[12][1]], [lmList[11][2], lmList[12][2]],
                        c='green')
                # Wrist-Ring
                ax.plot([lmList[0][0], lmList[13][0]], [lmList[0][1], lmList[13][1]], [lmList[0][2], lmList[13][2]],
                        c='red')
                ax.plot([lmList[13][0], lmList[14][0]], [lmList[13][1], lmList[14][1]], [lmList[13][2], lmList[14][2]],
                        c='yellow')
                ax.plot([lmList[14][0], lmList[15][0]], [lmList[14][1], lmList[15][1]], [lmList[14][2], lmList[15][2]],
                        c='yellow')
                ax.plot([lmList[15][0], lmList[16][0]], [lmList[15][1], lmList[16][1]], [lmList[15][2], lmList[16][2]],
                        c='yellow')
                # Wrist-Pinky
                ax.plot([lmList[0][0], lmList[17][0]], [lmList[0][1], lmList[17][1]], [lmList[0][2], lmList[17][2]],
                        c='red')
                ax.plot([lmList[17][0], lmList[18][0]], [lmList[17][1], lmList[18][1]], [lmList[17][2], lmList[18][2]],
                        c='blue')
                ax.plot([lmList[18][0], lmList[19][0]], [lmList[18][1], lmList[19][1]], [lmList[18][2], lmList[19][2]],
                        c='blue')
                ax.plot([lmList[19][0], lmList[20][0]], [lmList[19][1], lmList[20][1]], [lmList[19][2], lmList[20][2]],
                        c='blue')
                # plt.title(idx + 1)
                spacing = 0.2
                fig.subplots_adjust(bottom=spacing)

                plt.show()

    @staticmethod
    def distance_normalization(hand_list):
        total_distance = 0
        for coordinate_list in hand_list:
            temp = []
            counter = 0
            restructured_coord_list = []

            for coordinate in coordinate_list:
                temp.append(coordinate)
                if counter == 2:
                    restructured_coord_list.append(temp)
                    counter = 0
                    temp = []
                else:
                    counter = counter + 1

            distance1 = np.sqrt(
                np.sum(np.square(np.array(restructured_coord_list[5]) - np.array(restructured_coord_list[0]))))
            distance2 = np.sqrt(
                np.sum(np.square(np.array(restructured_coord_list[9]) - np.array(restructured_coord_list[0]))))
            distance3 = np.sqrt(
                np.sum(np.square(np.array(restructured_coord_list[13]) - np.array(restructured_coord_list[0]))))
            distance4 = np.sqrt(
                np.sum(np.square(np.array(restructured_coord_list[17]) - np.array(restructured_coord_list[0]))))
            total_distance = total_distance + distance1 + distance2 + distance3 + distance4

        scale_value = total_distance / (4 * len(hand_list))

        return scale_value

    @staticmethod
    def data_normalization(coordinates, min_ele, max_ele):

        non_zeros = np.count_nonzero(coordinates)
        if non_zeros != 0:
            for idx, element in enumerate(coordinates):
                coordinates[idx] = (element - min_ele) / (max_ele - min_ele)

        return coordinates

    @staticmethod
    def coordinate_regularization(coordinate_list):
        # print(type(coordinate_list))
        x = coordinate_list[0]
        y = coordinate_list[1]
        z = coordinate_list[2]
        reg_list = [x, y, z]
        temp = []
        counter = 0
        new_list = []
        for coord in coordinate_list:
            temp.append(coord)
            if counter == 2:
                difference = []
                zip_object = zip(temp, reg_list)
                for list1_i, list2_i in zip_object:
                    difference.append(list1_i - list2_i)
                counter = 0
                new_list = new_list + difference
                temp = []
            else:
                counter = counter + 1
        # print(type(new_list))
        return new_list

    @staticmethod
    def extract_hand_coordinates(hand_list, max_sequence_len):
        none_hand = False
        extend_seq = False
        all_frames = []
        for hand in hand_list:
            hands = []

            if hand != None:
                lmList = hand["lmList"]
                for landmark in lmList:
                    for coordinate in landmark:
                        hands.append(coordinate)
            else:
                none_hand = True
                hands = [0] * 63
            all_frames.append(hands)

        return all_frames
