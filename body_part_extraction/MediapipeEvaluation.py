import collections
from colorama import Fore, Style
import cv2
import os
from MyMediapipeToolkit import HandDetector
from HandSequenceToolkit import HandSequenceToolkit
from GrounfTruthBoxesToolkit import HandAndBoundingBoxesTools
from MyOpenPoseToolkit import OpenPoseModule
import matplotlib.pyplot as plt
import pandas as pd

def check_openpose_detections_to_help_mediapipe(root, og_img):
    local_open_pose_mod = OpenPoseModule()
    people_in_frames = []
    people_info_in_seq = []
    # img2 = cv2.imread("black_image.jpg")
    for idx, (root, dirs, files) in enumerate(os.walk(root)):
        for file in files:
            file_name = root + '\\' + file
            #file_name, hand_bboxes, signing_hands = metricsModule.extract_info_from_bbox_file(root, file)
            json = local_open_pose_mod.extract_json_from_dataset(file_name)
            all_hands, people_info, img2, people_in_picture, more_than_one_people, body_pos = local_open_pose_mod.extract_hand_data(og_img, json)
            people_info_in_seq.append(people_info)
            people_in_picture = local_open_pose_mod.extract_number_of_people(json)
            people_in_frames.append(people_in_picture)

    occurencies = collections.Counter(people_in_frames)
    max_key = max(occurencies, key=occurencies.get)

    return max_key

def draw_hand_connections(lmList, img, color):
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


def openpose_and_mediapipe_decision_making(open_pose_tools, hands, file_name, og_img, body_position, img, metrics_module, openpose_helped):
    json = open_pose_tools.extract_json_from_dataset(file_name)

    all_hands, people_info, openpose_img, people_in_picture, more_than_one_people, body_pos = open_pose_tools.extract_hand_data(
        og_img, json)

    if body_position == (0, 0, 0, 0):
        hands = all_hands
        if hands[0] != None:
            draw_hand_connections(hands[0]["lmList"], img, (255, 0, 0))
        if hands[1] != None:
            draw_hand_connections(hands[1]["lmList"], img, (0, 0, 0))
    else:
        openpose_mediapipe_disagree = True
        for person_info in people_info:
            pos = person_info["BodyPosition"]
            #print(pos)
            #print(body_position)
            iou = metrics_module.calculate_IOU_2(
                (pos[0], pos[1], (pos[0] + pos[2]), (pos[1] + pos[3])), (
                    body_position[0], body_position[1], (body_position[0] + body_position[2]),
                    (body_position[1] + body_position[3])))

            print(Fore.RED + f"IOU between bodies= {iou}")
            if iou > 0.4 and person_info["chosen"] == 1:
                openpose_mediapipe_disagree = False
                print(Fore.GREEN + "Openpose and mediapipe agree")

        if openpose_mediapipe_disagree:
            openpose_helped = openpose_helped + 1
            print(Fore.BLUE + "Openpose gets to choose the hands")
            hands = all_hands
            if hands[0] != None:
                draw_hand_connections(hands[0]["lmList"], img, (255, 0, 0))
            if hands[1] != None:
                draw_hand_connections(hands[1]["lmList"], img, (0, 0, 0))

    print(Style.RESET_ALL)

    return hands, img, openpose_helped


def main_mediapipe(path, write_to_excel=False, iou_mode=True, collaboration_mode=False):

    detection_con = 0.5
    min_track_con = 0.5
    mode = False
    rows = []
    hand_sequence = []
    img_sequence = []
    hands_detected_in_seq = []
    ground_truth_bbox_seq = []
    all_ground_truth_box_sequences = []
    metrics_module = HandAndBoundingBoxesTools()
    all_updated_hand_sequences = []
    open_pose_mod = OpenPoseModule()

    for idx1 in range(1):
        columns = []
        for idx2 in range(1):
            updater_iou = 0
            print(detection_con + (idx1 / 10))
            print(min_track_con + (idx2 / 10))
            detector = HandDetector(mode, 2, detection_con + (idx1 / 10), min_track_con + (idx2 / 10))
            total_iou = 0  # Total IOU score of the batch
            total_iou_updated = 0
            both_hands_sign = 0
            total_hands = 0  # Total number of hands in ground truth boxes
            total_signing_hands = 0  # Total number of signing hands in the ground truth boxes
            detected_signing_hands = 0  # Number of detected signing hands
            openpose_helped = 0
            prev_root = path
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            signing_hands = 0
            #bboxes_lens = []
            total_signing_hands_current = 0
            sequence_start = True
            people_in_video = 0

            for idx, (root, dirs, files) in enumerate(os.walk(path)):
                for file in files:

                    file_name = root + '\\' + file

                    hand_bboxes, signing_hands = metrics_module.extract_info_from_bbox_file(root, file_name)

                    img = metrics_module.extract_img_from_dataset(file_name)
                    og_img = metrics_module.extract_img_from_dataset(file_name)

                    if root.count('\\') == 6:
                        # Whenever we change a folder we have a new sequence
                        if prev_root != root:
                            sequence_start = True
                            # Before we update our sequence lists we perform an analysis and update of the previous sequence
                            updated_hand_sequence = []
                            if len(img_sequence) != 0 and len(hand_sequence) != 0:
                                editor = HandSequenceToolkit(img_sequence, hand_sequence, ground_truth_bbox_seq)
                                updated_hand_sequence,signing_hand_sequence, hand_info, up_iou, t_pos, f_pos, f_neg = editor.update_sequences(iou_mode)
                                if iou_mode:
                                    true_positives = true_positives + t_pos
                                    false_positives = false_positives + f_pos
                                    false_negatives = false_negatives + f_neg
                                    updater_iou = updater_iou + up_iou
                                    recall_cur = t_pos / total_signing_hands_current

                                    if recall_cur < 0.6:
                                        print(Fore.RED + "True positives = {}, False positives = {}, False Negatives = {}".format(t_pos, f_pos, f_neg))
                                        print(Style.RESET_ALL)
                                    print("In the current video Recall = {}".format(recall_cur))
                                    total_signing_hands_current = 0
                                all_updated_hand_sequences.append(updated_hand_sequence)
                                all_ground_truth_box_sequences.append(ground_truth_bbox_seq)
                                #bboxes_lens = []

                            print("root{}".format(prev_root))
                            ground_truth_bbox_seq = []
                            hand_sequence = []
                            img_sequence = []
                            people_in_frames = []
                            prev_root = root
                            people_in_video = 0

                            # This black image helps to reset the detector with every new sequence so old trends won't interfere with the result
                            img2 = cv2.imread("black_image.jpg")
                            detector.find_hands(img2, path=root, image_name=file_name, create_jsons=False)

                    if sequence_start  and collaboration_mode: #and root.count('\\') == 6
                        people_in_video = check_openpose_detections_to_help_mediapipe(root, og_img)
                        print(f"People in frames list {people_in_video}")
                        sequence_start = False

                    img, hands, body_position, _ = detector.find_hands(img, path=root, image_name=file_name, create_jsons=False)
                    #plt.imshow(img)
                    #plt.show()

                    if people_in_video > 1 and collaboration_mode:
                        print("Here",file_name)
                        hands, img, openpose_helped = openpose_and_mediapipe_decision_making(open_pose_mod, hands, file_name, og_img, body_position, img, metrics_module, openpose_helped)

                    #img = metricsModule.draw_bounding_boxes(img, hand_bboxes, hands)

                    #plt.imshow(img)
                    #plt.show()
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #cv2.imwrite('bbox_image.jpg', img)
                    # Append the newly acquired hands to a list for further temporal analysis

                    hand_sequence.append(hands)
                    img_sequence.append(img)
                    ground_truth_bbox_seq.append(hand_bboxes)
                    #bboxes_lens.append(len(hand_bboxes))

                    if signing_hands == 2:
                        both_hands_sign = both_hands_sign + 1
                        total_signing_hands = total_signing_hands + signing_hands - 1
                        total_signing_hands_current = total_signing_hands_current + signing_hands - 1
                    else:
                        total_signing_hands = total_signing_hands + signing_hands
                        total_signing_hands_current = total_signing_hands_current + signing_hands
                # total_iou = total_iou + iou

                # detected_signing_hands = detected_signing_hands + signing_hand_detected
            if len(hands_detected_in_seq) != 0 and min(hands_detected_in_seq) == 0:
                num_of_zero_hands = hands_detected_in_seq.count(0)
                percent = (num_of_zero_hands / len(hands_detected_in_seq)) * 100
                #sequences_with_zero_hands.append((root, percent))

            # Update the last sequence
            print("root{}".format(prev_root))
            editor = HandSequenceToolkit(img_sequence, hand_sequence, ground_truth_bbox_seq)
            updated_hand_sequence,signing_hand_sequence, hand_info, up_iou, t_pos, f_pos, f_neg = editor.update_sequences(iou_mode)

            true_positives = true_positives + t_pos
            false_positives = false_positives + f_pos
            false_negatives = false_negatives + f_neg
            updater_iou = updater_iou + up_iou
            print("True positives = {}, False positives = {}, False Negatives = {}".format(t_pos, f_pos, f_neg))
            recall_cur = t_pos / total_signing_hands_current
            if recall_cur < 0.6:
                print(Fore.RED + "True positives = {}, False positives = {}, False Negatives = {}".format(t_pos, f_pos,
                                                                                                          f_neg))
                print(Style.RESET_ALL)
            # print(t_pos)
            # print(f_pos)
            all_updated_hand_sequences.append(updated_hand_sequence)
            all_ground_truth_box_sequences.append(ground_truth_bbox_seq)
            ground_truth_bbox_seq = []
            hand_sequence = []
            img_sequence = []
            people_in_frames = []

            print("True positives = {}, False positives = {}, False Negatives = {}".format(true_positives,false_positives,false_negatives))
            if true_positives == 0 and false_positives == 0:
                precision = 0
            else:
                precision = true_positives / (true_positives + false_positives)
            recall = true_positives / total_signing_hands
            if precision == 0 and recall == 0:
                f1_score = 0
            else:
                f1_score = 2 * ((precision * recall) / (precision + recall))

            print(f"New way IOU = {updater_iou}")
            print(f"Precision = {precision}")
            print(f"Recall = {recall}")
            print(f"F1 Score = {f1_score}")
            print(f"Total signing hands = {total_signing_hands}")
            print(f"Both signing hands{both_hands_sign}")
            print(f"All occasions OpenPose helped{openpose_helped}")

            percent_of_detected_signing_hands = (detected_signing_hands / total_signing_hands) * 100

            # columns.append(iou_percentage)
            columns.append(updater_iou)
            columns.append(precision)
            columns.append(recall)
            columns.append(f1_score)
            # columns.append(percent_of_detected_signing_hands)
            columns.append("===")

        rows.append(columns)
    print(rows)
    if write_to_excel:
        df = pd.DataFrame(rows)
        df.to_excel(
            r'D:\University\ChicagoFSWild\ChicagoFSWild\Mediapipe_folder\Mediapipe_IOU_Recall_Precission_F1_v1.xlsx',
            sheet_name='new_sheet_name', index=False, header=False)


if __name__ == "__main__":
    ground_truth_path = r'D:\University\ChicagoFSWild\ChicagoFSWild\BBox'
    raw_data_path = r'D:\University\ChicagoFSWild\ChicagoFSWild\ChicagoFSWild-Frames'
    main_mediapipe(ground_truth_path, write_to_excel=False, iou_mode=True, collaboration_mode=True)

###############################CODE I MIGHT USE########################################


# Recalculate IOU after the update of the sequences
#            metricsModule = handAndGTBsMetrics()
#            for idx,sequence in enumerate(all_updated_hand_sequences):
#                updated_iou = recalculate_iou_after_sequence_update(img, metricsModule, sequence,all_ground_truth_box_sequences[idx])
#                total_iou_updated = total_iou_updated + updated_iou

# def recalculate_iou_after_sequence_update(img,metricsModule,hand_sequence,ground_truth_boxes):
#    total_updated_iou = 0

#    for idx,bboxes in enumerate(ground_truth_boxes):
#        temp = []
#       for hand in hand_sequence[idx]:
#            if hand != None:
#                temp.append(hand)
#        for ground_truth_box in bboxes:
#            IOU = metricsModule.calculate_IOU_between_hands_gtb_2(img, temp, ground_truth_box,False)
#            if ground_truth_box[4] == 1:
#                total_updated_iou = total_updated_iou + IOU
#    return total_updated_iou


#    def compare_hands_with_gtbs_after_update(self, img,hands,metricsModule,ground_truth_boxes):
#
#        total_iou = 0
#        temp = []
#        for hand in hands:
#            if hand != None:
#                temp.append(hand)

#        for idx, ground_truth_box in enumerate(ground_truth_boxes):
#            # print(ground_truth_box[4])
#            iou = metricsModule.calculate_IOU_between_hands_gtb_2(img, temp, ground_truth_box,False)
#            if ground_truth_box[4] == 1:
#                total_iou = total_iou + iou
#        return total_iou

# OPENPOSE PART
# json = OpenPoseMod.extract_json_from_dataset(file_name)
# all_hands, people_info, openpose_img, people_in_picture, more_than_one_people,body_pos = OpenPoseMod.extract_hand_data(img, json)
# people_in_frames.append(people_in_picture)
# # #Find hands and compare them with the ground truth bounding boxes
# img,hands,body_position = detector.findHands(img)
# if people_in_picture > 1:
#     if len(people_in_frames) != 0:
#         distance = abs(body_pos-body_position)
#         print(f"Mediapipe and OpenPose Body Pos difference {distance}")
#         cv2.circle(img, (0, int(body_position)), 3, (182, 10, 100), cv2.FILLED)
#         cv2.circle(img, (0, int(body_pos)), 3, (0, 255, 255), cv2.FILLED)
#         # temp_list = []
#         # for person_info in people_info:
#         #     pos = person_info["BodyPosition"]
#         #     print(f"OpenPose Body Pos {pos}")
#         #     distance = abs(body_position-person_info["BodyPosition"])
#         #     temp_list.append((person_info["chosen"], distance))
#         # sequence_start = False
#         # min_distance_tuple = min(temp_list, key=lambda t: t[1])
#         #print(f"Mediapipe detection closer to {min_distance_tuple[0]}")
#         plt.imshow(img)
#         plt.show()


# discrepancy detector
# if idx == 0:
#     prev_chosen_distance = chosen_distance[0]
# else:
#     all_positions = [dictionary["BodyPosition"] for dictionary in person_info if dictionary["chosen"] == 0]
#     all_positions.append(chosen_distance[0])
#     # print(all_positions)
#     all_positions[:] = [abs(pos - prev_chosen_distance) for pos in all_positions]
#     # print(all_positions)
#     minpos = all_positions.index(min(all_positions))
#     # print(minpos)
#     # print(len(all_positions))
#     temp_list.append(chosen_distance)
#     if minpos != len(all_positions) - 1:
#         discrepancy_check = True
#         discrepancy_list.append(discrepancy_value)
#         discrepancy_value = discrepancy_value + 1
#         chosen_body_positions_in_seq.append(temp_list)
#         temp_list = []
#     else:
#         discrepancy_list.append(discrepancy_value)
#         # chosen_body_positions_in_seq.append(temp_list)
#     prev_chosen_distance = chosen_distance[0]


# discrepancy_list = []
# discrepancy_value = 0
# temp_list = []
# chosen_body_positions_in_seq = []
# id_list = []
# discrepancy_check = False
# if max_key >= 2:
#     chosen_distance = 0
#     prev_chosen_distance = 0
#     for idx,person_info in enumerate(people_info_in_seq):
#         chosen_distance = [dictionary["BodyPosition"] for dictionary in person_info if dictionary["chosen"] == 1]
#         id = [dictionary["id"] for dictionary in person_info if dictionary["chosen"] == 1]
#         id_list.append(id[0])
#         if idx == 0:
#              prev_chosen_distance = chosen_distance[0]
#         else:
#             all_positions = [dictionary["BodyPosition"] for dictionary in person_info if dictionary["chosen"] == 0]
#             all_positions.append(chosen_distance[0])
#             # print(all_positions)
#             all_positions[:] = [abs(pos - prev_chosen_distance) for pos in all_positions]
#             # print(all_positions)
#             minpos = all_positions.index(min(all_positions))
#             # print(minpos)
#             # print(len(all_positions))
#             temp_list.append(chosen_distance)
#             if minpos != len(all_positions) - 1:
#                 discrepancy_check = True
#                 discrepancy_list.append(discrepancy_value)
#                 discrepancy_value = discrepancy_value + 1
#                 chosen_body_positions_in_seq.append(temp_list)
#                 temp_list = []
#             else:
#                 discrepancy_list.append(discrepancy_value)
#                 # chosen_body_positions_in_seq.append(temp_list)
#             prev_chosen_distance = chosen_distance[0]
#
# if discrepancy_check:
#     print(Fore.RED + "Chosen person changes in sequence")
#     print(discrepancy_list)
#     print(Style.RESET_ALL)

# print(f"Most probable number of people is {max_key}")

# Inside OpenPose and Mediapipe coordination function

# if len(people_in_frames) != 0:
# distance = abs(body_pos-body_position)
# print(f"Mediapipe body pos{body_position}")
# print(f"OpenPose body pos{body_pos}")
# print(f"Mediapipe and OpenPose Body Pos difference {distance}")
# cv2.circle(img, (0, int(body_position)), 3, (182, 10, 100), cv2.FILLED)
# cv2.circle(img, (0, int(body_pos)), 3, (0, 255, 255), cv2.FILLED)
# temp_list = []
