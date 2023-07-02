import os
import cv2
from GrounfTruthBoxesToolkit import HandAndBoundingBoxesTools
from HandSequenceToolkit import HandSequenceToolkit
import matplotlib.pyplot as plt
from MyOpenPoseToolkit import OpenPoseModule


def main_openpose():
    path = r'D:\University\ChicagoFSWild\ChicagoFSWild\BBox'
    final_iou = 0
    total_signing_hands = 0
    hand_sequence = []
    people_sequence = []
    img_sequence = []
    ground_truth_bbox_seq = []
    all_updated_hand_sequences = []
    all_ground_truth_box_sequences = []
    updater_iou = 0
    prev_root = path
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    metrics_module = HandAndBoundingBoxesTools()
    images_with_multiple_people = 0
    new_seq = True
    open_pose_mod = OpenPoseModule()
    for idx, (root, dirs, files) in enumerate(os.walk(path)):
        for file in files:

            image_file_name = root + '\\' + file
            hand_bboxes, signing_hands = metrics_module.extract_info_from_bbox_file(root, image_file_name)
            img = metrics_module.extract_img_from_dataset(image_file_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if root.count('\\') == 6:
                if prev_root != root:
                    if len(img_sequence) != 0 and len(hand_sequence) != 0:
                        editor = HandSequenceToolkit(img_sequence, hand_sequence, ground_truth_bbox_seq)
                        updated_hand_sequence, signing_hand_sequence, hand_info, up_iou, t_pos, f_pos, f_neg = editor.update_sequences(True)

                        true_positives = true_positives + t_pos
                        false_positives = false_positives + f_pos
                        false_negatives = false_negatives + f_neg
                        updater_iou = updater_iou + up_iou
                    new_seq = True
                    ground_truth_bbox_seq = []
                    hand_sequence = []
                    img_sequence = []
                    print("root{}".format(prev_root))
                    print("")
                    prev_root = root

            json = open_pose_mod.extract_json_from_dataset(image_file_name)
            all_hands, people_info, img, people_in_picture, more_than_one_people, body_pos = open_pose_mod.extract_hand_data(
                img, json)
            if more_than_one_people:
                images_with_multiple_people = images_with_multiple_people + 1

            people_sequence.append(people_info)
            hand_sequence.append(all_hands)

            ground_truth_bbox_seq.append(hand_bboxes)
            img = cv2.resize(img, (640, 360))

            img = metrics_module.draw_bounding_boxes(img, hand_bboxes, all_hands)
            print("Hello")
            img = open_pose_mod.draw_hand_landmarks(all_hands, img)
            plt.imshow(img)
            plt.show()
            # save = input('Save?:')
            # if save == 'y':
            #      cv2.imwrite("low_visibility_keypoints.jpeg", img)
            img_sequence.append(img)

            total_signing_hands = total_signing_hands + signing_hands

    editor = HandSequenceToolkit(img_sequence, hand_sequence, ground_truth_bbox_seq)
    updated_hand_sequence, signing_hand_sequence, hand_info, up_iou, t_pos, f_pos, f_neg = editor.update_sequences(True)
    updater_iou = updater_iou + up_iou
    true_positives = true_positives + t_pos
    false_positives = false_positives + f_pos
    false_negatives = false_negatives + f_neg

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_signing_hands
    print(f"New way IOU = {updater_iou}")
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"Total signing hands = {total_signing_hands}")
    print(images_with_multiple_people)


if __name__ == "__main__":
    main_openpose()


