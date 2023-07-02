from colorama import Fore, Style
import cv2
import os
from MyMediapipeToolkit import HandDetector
from HandSequenceToolkit import HandSequenceToolkit
from GrounfTruthBoxesToolkit import HandAndBoundingBoxesTools
from MyOpenPoseToolkit import OpenPoseModule
import pandas as pd
import numpy as np
import collections
import pickle
from numpy import save
import json
open_pose_mod = OpenPoseModule()
metrics_module = HandAndBoundingBoxesTools()


def check_openpose_detections_to_help_mediapipe(root, og_img):
    people_in_frames = []
    people_info_in_seq = []

    for file in os.listdir(root):
        file_name = root + '\\' + file
        all_hands, people_info, img2, people_in_picture, more_than_one_people, body_pos = open_pose_mod.extract_hand_data(
            og_img, file_name)
        people_info_in_seq.append(people_info)
        people_in_picture = open_pose_mod.extract_number_of_people(file_name)
        people_in_frames.append(people_in_picture)

    occurrences = collections.Counter(people_in_frames)
    if len(occurrences) == 0:
        max_key = 1
    else:
        max_key = max(occurrences, key=occurrences.get)

    return max_key


def openpose_and_mediapipe_decision_making(hands, json_file, og_img, body_position):
    assist = False
    all_hands, people_info, openpose_img, people_in_picture, more_than_one_people, body_pos = open_pose_mod.extract_hand_data(
        og_img, json_file)

    if body_position == (0, 0, 0, 0):
        hands = all_hands
        assist = True
    else:
        openpose_mediapipe_disagree = True
        for person_info in people_info:
            pos = person_info["BodyPosition"]
            iou = metrics_module.calculate_IOU_2(
                (pos[0], pos[1], (pos[0] + pos[2]), (pos[1] + pos[3])), (
                    body_position[0], body_position[1], (body_position[0] + body_position[2]),
                    (body_position[1] + body_position[3])))

            print(Fore.RED + f"IOU between bodies= {iou}")
            if iou > 0.1 and person_info["chosen"] == 1:
                openpose_mediapipe_disagree = False
                assist = False
                print(Fore.GREEN + "Openpose and mediapipe agree")

        if openpose_mediapipe_disagree:
            print(Fore.BLUE + "Openpose gets to choose the hands")
            hands = all_hands
            assist = True

    print(Style.RESET_ALL)

    return hands, assist


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
        all_frames.append(np.array(hands))

    # if none_hand:
    #     print(all_frames)
    if extend_seq:
        if len(hand_list) < max_sequence_len:
            hands = [0] * 63
            for i in range(max_sequence_len - len(hand_list)):
                all_frames.append(np.array(hands))
    all_frames = np.array(all_frames)
    return all_frames


def extract_bodypose_coordinates(body_keypoints):
    all_frames = []
    for body in body_keypoints:
        body_pose = []

        for body_coordinate in body:
            for idx, keypoint in enumerate(body_coordinate[:3]):
                if idx == 0 and keypoint > 640:
                    break
                if idx == 1 and keypoint > 360:
                    break
                body_pose.append(keypoint)

        if len(body_pose) < 99:
            print(f"Missing {99 - len(body_pose)} keypoints in body")
            body_pose.extend([0] * (99 - len(body_pose)))
        if len(body_pose) > 99:
            print(len(body_pose))
            print("Wrong")
        all_frames.append(np.array(body_pose))
    return all_frames


def max_min_words_in_sentence(word_list):
    num_of_words = [len(sentence.split()) for sentence in word_list]
    return max(num_of_words), min(num_of_words)


def max_string(word_list):
    string_lengths = [len(sentence) for idx, sentence in enumerate(word_list)]
    return max(string_lengths)


def extract_letters_from_labels(word_list):
    print(word_list)
    words_counter = collections.Counter([word for sentence in word_list for word in sentence.split()])
    print('{} English words.'.format(len([word for sentence in word_list for word in sentence.split()])))
    print('{} unique English words.'.format(len(words_counter)))
    print('10 Most common words in the English dataset:')
    max_sentence, min_sentence = max_min_words_in_sentence(word_list)
    max_string_length = max_string(word_list)
    print(words_counter)
    print("Longest sentence is {} and shortest is {}".format(max_sentence, min_sentence))
    print('"' + '" "'.join(list(zip(*words_counter.most_common(10)))[0]) + '"')
    print(words_counter.most_common(10))
    all_sentences_tokenized = []
    for sentence in word_list:
        video_sentence = [0] * max_string_length
        for idx, character in enumerate(sentence):
            if character == ' ':
                numeric_value = 0
            else:
                numeric_value = ord(character) - 96
            video_sentence[idx] = numeric_value
        all_sentences_tokenized.append(np.array(video_sentence))

    return all_sentences_tokenized


def collect_signing_right_left_hand_datapoints_from_mediapipe(csv_path, frame_path):
    train_set_data = []
    train_set_data_right_hands = []
    train_set_data_left_hands = []
    train_set_data_body = []
    train_set_labels = []

    validation_set_data = []
    validation_set_data_right_hands = []
    validation_set_data_left_hands = []
    validation_set_data_body = []
    validation_set_labels = []

    test_set_data = []
    test_set_data_right_hands = []
    test_set_data_left_hands = []
    test_set_data_body = []
    test_set_labels = []

    detection_con = 0.5
    min_track_con = 0.5
    mode = False
    detector = HandDetector(mode, 2, detection_con, min_track_con)

    hand_bboxes = None
    ground_truth_bbox_seq = []
    iou_mode = False

    df = pd.read_csv(csv_path, usecols=['filename', 'number_of_frames', 'label_proc', 'partition'])
    sequence_lengths = df['number_of_frames']
    max_sequence_len = max(sequence_lengths)
    # detector = handDetector(mode, 2, detectionCon, minTrackCon)

    word_list = df['label_proc']
    all_sentences_tokenized = extract_letters_from_labels(word_list)
    df["tokenized_label_proc"] = all_sentences_tokenized
    img2 = cv2.imread("black_image.jpg")
    for idx in df.index:
        hand_sequence = []
        body_keypoints_sequence = []
        img_sequence = []
        video_path = os.path.join(frame_path, df["filename"][idx])

        for file_name in os.listdir(video_path):
            if file_name != 'json':
                image_file_name = os.path.join(video_path, file_name)
                img = metrics_module.extract_img_from_dataset(image_file_name)
                img, hands, body_position, body_keypoints = detector.find_hands(img, path=video_path, image_name=image_file_name,
                                                                create_jsons=False)
                img = metrics_module.draw_bounding_boxes(img, hand_bboxes, hands)
                hand_sequence.append(hands)
                body_keypoints_sequence.append(body_keypoints)
                img_sequence.append(img)

        detector.find_hands(img2, path=video_path, image_name=file_name, create_jsons=False)

        editor = HandSequenceToolkit(img_sequence, hand_sequence, ground_truth_bbox_seq)
        updated_hand_sequence, signing_hand_sequence, hand_info, up_iou, t_pos, f_pos, f_neg = editor.update_sequences(iou_mode)

        complete_signing_hand_sequence = extract_hand_coordinates(signing_hand_sequence, max_sequence_len)
        right_hand_sequence, left_hand_sequence = zip(*updated_hand_sequence)
        complete_right_hand_sequence = extract_hand_coordinates(right_hand_sequence, max_sequence_len)
        complete_left_hand_sequence = extract_hand_coordinates(left_hand_sequence, max_sequence_len)
        complete_body_pose_sequence = extract_bodypose_coordinates(body_keypoints_sequence)

        if df["partition"][idx] == 'train':
            train_set_data.append(complete_signing_hand_sequence)
            train_set_data_right_hands.append(complete_right_hand_sequence)
            train_set_data_left_hands.append(complete_left_hand_sequence)
            train_set_data_body.append(complete_body_pose_sequence)
            train_set_labels.append(df["tokenized_label_proc"][idx])
        if df["partition"][idx] == 'dev':
            validation_set_data.append(complete_signing_hand_sequence)
            validation_set_data_right_hands.append(complete_right_hand_sequence)
            validation_set_data_left_hands.append(complete_left_hand_sequence)
            validation_set_data_body.append(complete_body_pose_sequence)
            validation_set_labels.append(df["tokenized_label_proc"][idx])
        if df["partition"][idx] == 'test':
            test_set_data.append(complete_signing_hand_sequence)
            test_set_data_right_hands.append(complete_right_hand_sequence)
            test_set_data_left_hands.append(complete_left_hand_sequence)
            test_set_data_body.append(complete_body_pose_sequence)
            test_set_labels.append(df["tokenized_label_proc"][idx])
        print("Video path: {}, Completion percentage {:.2f}".format(video_path, (idx / df.index[-1]) * 100))

    train_set_data = np.array(train_set_data)
    train_set_data_right_hands = np.array(train_set_data_right_hands)
    train_set_data_left_hands = np.array(train_set_data_left_hands)
    train_set_data_body = np.array(train_set_data_body)
    train_set_labels = np.array(train_set_labels)
    print(train_set_data.shape)

    validation_set_data = np.array(validation_set_data)
    validation_set_data_right_hands = np.array(validation_set_data_right_hands)
    validation_set_data_left_hands = np.array(validation_set_data_left_hands)
    validation_set_data_body = np.array(validation_set_data_body)
    validation_set_labels = np.array(validation_set_labels)
    print(validation_set_data.shape)

    test_set_data = np.array(test_set_data)
    test_set_data_right_hands = np.array(test_set_data_right_hands)
    test_set_data_left_hands = np.array(test_set_data_left_hands)
    test_set_data_body = np.array(test_set_data_body)
    test_set_labels = np.array(test_set_labels)
    print(test_set_data.shape)

    try:
        save('../mediapipe_data/train_set_data_signing.npy', train_set_data)
        save('../mediapipe_data/train_set_data_right.npy', train_set_data_right_hands)
        save('../mediapipe_data/train_set_data_left.npy', train_set_data_left_hands)
        save('../mediapipe_data/train_set_data_left.npy', train_set_data_body)
        save('../mediapipe_data/train_set_labels.npy', train_set_labels)

        save('../mediapipe_data/validation_set_data_signing.npy', validation_set_data)
        save('../mediapipe_data/val_set_data_right.npy', validation_set_data_right_hands)
        save('../mediapipe_data/val_set_data_left.npy', validation_set_data_left_hands)
        save('../mediapipe_data/val_set_data_left.npy', validation_set_data_body)
        save('../mediapipe_data/validation_set_labels.npy', validation_set_labels)

        save('../mediapipe_data/test_set_data_signing.npy', test_set_data)
        save('../mediapipe_data/test_set_data_right.npy', test_set_data_right_hands)
        save('../mediapipe_data/test_set_data_left.npy', test_set_data_left_hands)
        save('../mediapipe_data/val_set_data_left.npy', test_set_data_body)
        save('../mediapipe_data/validation_set_labels.npy', test_set_labels)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def collect_signing_right_left_hand_datapoints_openpose_and_mediapipe(csv_path, frame_path):
    train_set_data = []
    train_set_data_right_hands = []
    train_set_data_left_hands = []
    train_set_labels = []

    validation_set_data = []
    validation_set_data_right_hands = []
    validation_set_data_left_hands = []
    validation_set_labels = []

    test_set_data = []
    test_set_data_right_hands = []
    test_set_data_left_hands = []
    test_set_labels = []

    # directory = r"D:\University\ChicagoFSWild\ChicagoFSWild\MP&OP_hand_images"
    df = pd.read_csv(csv_path, usecols=['filename', 'number_of_frames', 'label_proc', 'partition'])
    ground_truth_bbox_seq = []


    detection_con = 0.5
    min_track_con = 0.5
    mode = False
    detector = HandDetector(mode, 2, detection_con, min_track_con)
    assists = []
    hand_bboxes = None

    word_list = df['label_proc']
    all_sentences_tokenized = extract_letters_from_labels(word_list)
    df["tokenized_label_proc"] = all_sentences_tokenized

    for idx in df.index:

        hand_sequence = []
        img_sequence = []

        video = df["filename"][idx]
        video_path = os.path.join(frame_path, video)
        jsons_path = os.path.join(video_path, "json")

        openpose_video_path = os.path.join("D:\ChicagoFSWildFramesOpenpose", df["filename"][idx], "json_files")

        openpose_helped = False
        for file_name in os.listdir(video_path):
            if file_name != 'json':
                image_file_name = os.path.join(video_path, file_name)
                img = metrics_module.extract_img_from_dataset(image_file_name)
                img, hands, body_position, _ = detector.find_hands(img, path=video_path, image_name=image_file_name,
                                                                create_jsons=False)
                img = metrics_module.draw_bounding_boxes(img, hand_bboxes, hands)
                if os.path.isdir(openpose_video_path):
                    people_in_video = check_openpose_detections_to_help_mediapipe(openpose_video_path, img)
                else:
                    print(Fore.RED + "No json from OpenPose")
                    print(openpose_video_path)
                    print(Style.RESET_ALL)
                    people_in_video = 1
                print(f"People in video {people_in_video}")

                if people_in_video > 1:
                    json_file_name = os.path.join(openpose_video_path, file_name.replace(".jpg", "_keypoints.json"))
                    hands, openpose_helped = openpose_and_mediapipe_decision_making(hands, json_file_name, img,
                                                                                    body_position)
                assists.append(openpose_helped)
                img_sequence.append(img)
                hand_sequence.append(hands)

        editor = HandSequenceToolkit(img_sequence, hand_sequence, ground_truth_bbox_seq)
        updated_hand_sequence, signing_hand_sequence, hand_info, up_iou, t_pos, f_pos, f_neg = editor.update_sequences(
            False)
        complete_hand_sequence = extract_hand_coordinates(signing_hand_sequence, 0)

        right_hand_sequence, left_hand_sequence = zip(*updated_hand_sequence)
        complete_right_hand_sequence = extract_hand_coordinates(right_hand_sequence, 0)
        complete_left_hand_sequence = extract_hand_coordinates(left_hand_sequence, 0)

        if df["partition"][idx] == 'train':
            train_set_data.append(complete_hand_sequence)
            train_set_data_right_hands.append(complete_right_hand_sequence)
            train_set_data_left_hands.append(complete_left_hand_sequence)
            train_set_labels.append(df["tokenized_label_proc"][idx])
        if df["partition"][idx] == 'dev':
            validation_set_data.append(complete_hand_sequence)
            validation_set_data_right_hands.append(complete_right_hand_sequence)
            validation_set_data_left_hands.append(complete_left_hand_sequence)
            validation_set_labels.append(df["tokenized_label_proc"][idx])
        if df["partition"][idx] == 'test':
            test_set_data.append(complete_hand_sequence)
            test_set_data_right_hands.append(complete_right_hand_sequence)
            test_set_data_left_hands.append(complete_left_hand_sequence)
            test_set_labels.append(df["tokenized_label_proc"][idx])
        print("Video path: {}, Completion percentage {:.2f}".format(video_path, (idx / df.index[-1]) * 100))

    train_set_data = np.array(train_set_data)
    train_set_data_right_hands = np.array(train_set_data_right_hands)
    train_set_data_left_hands = np.array(train_set_data_left_hands)
    train_set_labels = np.array(train_set_labels)
    print(train_set_data.shape)

    validation_set_data = np.array(validation_set_data)
    validation_set_data_right_hands = np.array(validation_set_data_right_hands)
    validation_set_data_left_hands = np.array(validation_set_data_left_hands)
    validation_set_labels = np.array(validation_set_labels)
    print(validation_set_data.shape)

    test_set_data = np.array(test_set_data)
    test_set_data_right_hands = np.array(test_set_data_right_hands)
    test_set_data_left_hands = np.array(test_set_data_left_hands)
    test_set_labels = np.array(test_set_labels)
    print(test_set_data.shape)

    print(sum(assists))
    try:
        save('../mediapipe_and_openpose_data/train_set_data.npy', train_set_data)
        save('../mediapipe_and_openpose_data/train_set_data_right.npy', train_set_data_right_hands)
        save('../mediapipe_and_openpose_data/train_set_data_left.npy', train_set_data_left_hands)
        save('../mediapipe_and_openpose_data/train_set_labels.npy', train_set_labels)

        save('../mediapipe_and_openpose_data/validation_set_data.npy', validation_set_data)
        save('../mediapipe_and_openpose_data/val_set_data_right.npy', validation_set_data_right_hands)
        save('../mediapipe_and_openpose_data/val_set_data_left.npy', validation_set_data_left_hands)
        save('../mediapipe_and_openpose_data/validation_set_labels.npy', validation_set_labels)

        save('../mediapipe_and_openpose_data/test_set_data.npy', test_set_data)
        save('../mediapipe_and_openpose_data/test_set_data_right.npy', test_set_data_right_hands)
        save('../mediapipe_and_openpose_data/test_set_data_left.npy', test_set_data_left_hands)
        save('../mediapipe_and_openpose_data/test_set_labels.npy', test_set_labels)

    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def collect_signing_right_left_hand_datapoints_from_openpose(csv_path, frame_path):
    train_set_data_signing_hands = []
    train_set_data_right_hands = []
    train_set_data_left_hands = []
    train_set_data_body = []
    train_set_labels = []

    validation_set_data_signing_hands = []
    validation_set_data_right_hands = []
    validation_set_data_left_hands = []
    validation_set_data_body = []
    validation_set_labels = []

    test_set_data_signing_hands = []
    test_set_data_right_hands = []
    test_set_data_left_hands = []
    test_set_data_body = []
    test_set_labels = []


    df = pd.read_csv(csv_path, usecols=['filename', 'number_of_frames', 'label_proc', 'partition'])
    ground_truth_bbox_seq = []


    print("hello")
    print(len(df.index))
    for idx in df.index:

        hand_sequence = []
        img_sequence = []

        video = df["filename"][idx]
        video_path = os.path.join(frame_path, video)
        jsons_path = os.path.join(video_path, "json_files")

        if os.path.isdir(video_path) == False or os.path.isdir(jsons_path) == False:
            print(video)

        for image_file_name in os.listdir(video_path):
            if image_file_name != 'json' and image_file_name != 'json_files' and not "rendered" in image_file_name:
                image_file = os.path.join(video_path, image_file_name)
                img = cv2.imread(image_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = rescale_image(img)
                img_sequence.append(img)
        openpose_video_path = os.path.join("D:\ChicagoFSWildFramesOpenpose", df["filename"][idx], "json_files")
        # print(openpose_video_path)

        if os.path.isdir(jsons_path):
            pass
            # people_in_video = check_OpenPose_detections_to_help_MediaPipe(openpose_video_path,cv2.imread(image_file))
        else:
            print(Fore.RED + "No json from OpenPose")
            print(jsons_path)
            print(Style.RESET_ALL)
            people_in_video = 1
        if len(os.listdir(jsons_path)) == 0:
            print(Fore.GREEN + "Empty json folder from OpenPose")
            print(jsons_path)
            print(Style.RESET_ALL)
            # continue

        # print(f"People in video {people_in_video}")

        # openpose_helped = False
        for json_file_name in os.listdir(jsons_path):
            if not "rendered" in json_file_name:
                json_file = os.path.join(jsons_path, json_file_name)
                all_hands, people_info, openpose_img, people_in_picture, more_than_one_people, body_pos = open_pose_mod.extract_hand_data(
                    cv2.imread(image_file), json_file)
                hand_sequence.append(all_hands)
            else:
                print("I skipped a rendered JSON")

        editor = HandSequenceToolkit(img_sequence, hand_sequence, ground_truth_bbox_seq)
        updated_hand_sequence, signing_hand_sequence, hand_info, up_iou, t_pos, f_pos, f_neg = editor.update_sequences(
            False)

        right_hand_sequence, left_hand_sequence = zip(*updated_hand_sequence)

        complete_signing_hand_sequence = extract_hand_coordinates(signing_hand_sequence, 0)
        complete_right_hand_sequence = extract_hand_coordinates(right_hand_sequence, 0)
        complete_left_hand_sequence = extract_hand_coordinates(left_hand_sequence, 0)

        # complete_body_pose_sequence = extract_bodypose_coordinates(body_keypoints_sequence)

        if df["partition"][idx] == 'train':
            train_set_data_signing_hands.append(complete_signing_hand_sequence)
            train_set_data_right_hands.append(complete_right_hand_sequence)
            train_set_data_left_hands.append(complete_left_hand_sequence)
            # train_set_data_body.append(complete_body_pose_sequence)
            train_set_labels.append(df["tokenized_label_proc"][idx])
        if df["partition"][idx] == 'dev':
            validation_set_data_signing_hands.append(complete_signing_hand_sequence)
            validation_set_data_right_hands.append(complete_right_hand_sequence)
            validation_set_data_left_hands.append(complete_left_hand_sequence)
            # validation_set_data_body.append(complete_body_pose_sequence)
            validation_set_labels.append(df["tokenized_label_proc"][idx])
        if df["partition"][idx] == 'test':
            test_set_data_signing_hands.append(complete_signing_hand_sequence)
            test_set_data_right_hands.append(complete_right_hand_sequence)
            test_set_data_left_hands.append(complete_left_hand_sequence)
            # test_set_data_body.append(complete_body_pose_sequence)
            test_set_labels.append(df["tokenized_label_proc"][idx])
        print("Video path: {}, Completion percentage {:.2f}".format(video_path, (idx / df.index[-1]) * 100))

    train_set_data_signing_hands = np.array(train_set_data_signing_hands)
    train_set_data_right_hands = np.array(train_set_data_right_hands)
    train_set_data_left_hands = np.array(train_set_data_left_hands)
    # train_set_data_body = np.array(train_set_data_body)
    train_set_labels = np.array(train_set_labels)
    # print(train_set_data.shape)
    #
    validation_set_data_signing_hands = np.array(validation_set_data_signing_hands)
    validation_set_data_right_hands = np.array(validation_set_data_right_hands)
    validation_set_data_left_hands = np.array(validation_set_data_left_hands)
    # validation_set_data_body = np.array(validation_set_data_body)
    validation_set_labels = np.array(validation_set_labels)
    # print(validation_set_data.shape)

    test_set_data_signing_hands = np.array(test_set_data_signing_hands)
    test_set_data_right_hands = np.array(test_set_data_right_hands)
    test_set_data_left_hands = np.array(test_set_data_left_hands)
    # test_set_data_body = np.array(test_set_data_body)
    # test_set_data_body = np.array([np.array(x) for x in test_set_data_body],dtype=object)
    test_set_labels = np.array(test_set_labels)
    # print(test_set_data.shape)
    #
    try:
        save('../pytorch_tutorials/openpose_data/train_set_data_signing_hands.npy', train_set_data_signing_hands)
        save('../pytorch_tutorials/openpose_data/train_set_data_right_hands.npy', train_set_data_right_hands)
        save('../pytorch_tutorials/openpose_data/train_set_data_left_hands.npy', train_set_data_left_hands)
        # np.save('../pytorch_tutorials/openpose_data/train_set_data_body.npy', train_set_data_body)
        save('../pytorch_tutorials/train_set_labels.npy', train_set_labels)

        save('../pytorch_tutorials/openpose_data/validation_set_data_signing_hands.npy',
             validation_set_data_signing_hands)
        save('../pytorch_tutorials/openpose_data/validation_set_data_right_hands.npy', validation_set_data_right_hands)
        save('../pytorch_tutorials/openpose_data/validation_set_data_left_hands.npy', validation_set_data_left_hands)
        # np.save('../pytorch_tutorials/validation_set_data_body.npy', validation_set_data_body)
        save('../pytorch_tutorials/validation_set_labels.npy', validation_set_labels)

        save('../pytorch_tutorials/openpose_data/test_set_data_signing_hands.npy', test_set_data_signing_hands)
        save('../pytorch_tutorials/openpose_data/test_set_data_right_hands.npy', test_set_data_right_hands)
        save('../pytorch_tutorials/openpose_data/test_set_data_left_hands.npy', test_set_data_left_hands)
        # np.save('../pytorch_tutorials/test_set_data_body.npy', test_set_data_body)
        save('../pytorch_tutorials/test_set_labels.npy', test_set_labels)

    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


'####################### COULD BE OBSOLETE START ############################'


# def collect_all_body_part_datapoints_from_mediapipe(csv_path, frame_path):
#     train_set_data_right_hands = []
#     train_set_data_left_hands = []
#     train_set_data_body = []
#     train_set_labels = []
#
#     validation_set_data_right_hands = []
#     validation_set_data_left_hands = []
#     validation_set_data_body = []
#     validation_set_labels = []
#
#     test_set_data_right_hands = []
#     test_set_data_left_hands = []
#     test_set_data_body = []
#     test_set_labels = []
#
#     ground_truth_bbox_seq = []
#
#     df = pd.read_csv(csv_path, usecols=['filename', 'number_of_frames', 'label_proc', 'partition'])
#     sequence_lengths = df['number_of_frames']
#     max_sequence_len = max(sequence_lengths)
#     # detector = handDetector(mode, 2, detectionCon, minTrackCon)
#
#     word_list = df['label_proc']
#     all_sentences_tokenized = extract_letters_from_labels(word_list)
#     df["tokenized_label_proc"] = all_sentences_tokenized
#     img2 = cv2.imread("black_image.jpg")
#     for idx in df.index:
#         hand_sequence = []
#         body_keypoints_sequence = []
#         img_sequence = []
#         video_path = os.path.join(frame_path, df["filename"][idx])
#         jsons_path = os.path.join(video_path, "json")
#         print(jsons_path)
#         for json_file_name in os.listdir(jsons_path):
#             json_file = os.path.join(jsons_path, json_file_name)
#             all_hands, body_pos, poseLmList = extract_mediapipe_keypoints_from_JSON(json_file)
#             try:
#                 with open(json_file, "r") as signer_file:
#                     signer = json.loads(signer_file.read())
#             except json.decoder.JSONDecodeError:
#                 print("Error decoding JSON!")
#             hand_sequence.append(all_hands)
#             body_keypoints_sequence.append(signer["body_pose_keypoints"])
#
#         editor = handSequenceEditor(img_sequence, hand_sequence, ground_truth_bbox_seq)
#         updated_hand_sequence, signing_hand_sequence, hand_info, up_iou, t_pos, f_pos, f_neg = editor.update_sequences(
#             False)
#
#         # print(updated_hand_sequence)
#         right_hand_sequence, left_hand_sequence = zip(*updated_hand_sequence)
#         complete_right_hand_sequence = extract_hand_coordinates(right_hand_sequence, max_sequence_len)
#         complete_left_hand_sequence = extract_hand_coordinates(left_hand_sequence, max_sequence_len)
#         complete_body_pose_sequence = extract_bodypose_coordinates(body_keypoints_sequence)
#
#         if df["partition"][idx] == 'train':
#             train_set_data_right_hands.append(complete_right_hand_sequence)
#             train_set_data_left_hands.append(complete_left_hand_sequence)
#             train_set_data_body.append(complete_body_pose_sequence)
#             train_set_labels.append(df["tokenized_label_proc"][idx])
#         if df["partition"][idx] == 'dev':
#             validation_set_data_right_hands.append(complete_right_hand_sequence)
#             validation_set_data_left_hands.append(complete_left_hand_sequence)
#             validation_set_data_body.append(complete_body_pose_sequence)
#             validation_set_labels.append(df["tokenized_label_proc"][idx])
#         if df["partition"][idx] == 'test':
#             test_set_data_right_hands.append(complete_right_hand_sequence)
#             test_set_data_left_hands.append(complete_left_hand_sequence)
#             test_set_data_body.append(complete_body_pose_sequence)
#             test_set_labels.append(df["tokenized_label_proc"][idx])
#         print("Video path: {}, Completion percentage {:.2f}".format(video_path, (idx / df.index[-1]) * 100))
#
#     train_set_data_right_hands = np.array(train_set_data_right_hands)
#     train_set_data_left_hands = np.array(train_set_data_left_hands)
#     train_set_data_body = np.array(train_set_data_body)
#     # train_set_labels = np.array(train_set_labels)
#     # print(train_set_data.shape)
#     #
#     validation_set_data_right_hands = np.array(validation_set_data_right_hands)
#     validation_set_data_left_hands = np.array(validation_set_data_left_hands)
#     validation_set_data_body = np.array(validation_set_data_body)
#     # validation_set_labels = np.array(validation_set_labels)
#     # print(validation_set_data.shape)
#     #
#     test_set_data_right_hands = np.array(test_set_data_right_hands)
#     test_set_data_left_hands = np.array(test_set_data_left_hands)
#     test_set_data_body = np.array(test_set_data_body)
#     # test_set_data_body = np.array([np.array(x) for x in test_set_data_body],dtype=object)
#     test_set_labels = np.array(test_set_labels)
#     # print(test_set_data.shape)
#     #
#     try:
#         np.save('../pytorch_tutorials/train_set_data_right_hands.npy', train_set_data_right_hands)
#         np.save('../pytorch_tutorials/train_set_data_left_hands.npy', train_set_data_left_hands)
#         np.save('../pytorch_tutorials/train_set_data_body.npy', train_set_data_body)
#         # save('../pytorch_tutorials/train_set_labels.npy', train_set_labels)
#
#         np.save('../pytorch_tutorials/validation_set_data_right_hands.npy', validation_set_data_right_hands)
#         np.save('../pytorch_tutorials/validation_set_data_left_hands.npy', validation_set_data_left_hands)
#         np.save('../pytorch_tutorials/validation_set_data_body.npy', validation_set_data_body)
#         # save('../pytorch_tutorials/validation_set_labels.npy', validation_set_labels)
#
#         np.save('../pytorch_tutorials/test_set_data_right_hands.npy', test_set_data_right_hands)
#         np.save('../pytorch_tutorials/test_set_data_left_hands.npy', test_set_data_left_hands)
#         np.save('../pytorch_tutorials/test_set_data_body.npy', test_set_data_body)
#         # save('../pytorch_tutorials/test_set_labels.npy', test_set_labels)
#
#     except Exception as ex:
#         print("Error during pickling object (Possibly unsupported):", ex)


'####################### COULD BE OBSOLETE END ############################'


def rescale_image(img):
    img = cv2.resize(img, (640, 360))

    return img


def unpack_data(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def word_statistics(csv_path):
    df = pd.read_csv(csv_path, usecols=['filename', 'number_of_frames', 'label_proc', 'partition'])
    sequence_lengths = df['number_of_frames']
    max_sequence_len = max(sequence_lengths)
    print(f"Max seq len {max_sequence_len}")
    # detector = handDetector(mode, 2, detectionCon, minTrackCon)

    word_list = df['label_proc']
    all_sentences_tokenized = extract_letters_from_labels(word_list)


if __name__ == "__main__":
    language_statistics = False
    combined_effort = False
    mediapipe_data_extraction = True
    openpose_data_extraction = False

    frame_path = r"D:\University\ChicagoFSWild\ChicagoFSWild\ChicagoFSWild-Frames"
    csv_path = r"D:\University\ChicagoFSWild\ChicagoFSWild\ChicagoFSWild.csv"

    if language_statistics:
        word_statistics(csv_path)
    if combined_effort:
        collect_signing_right_left_hand_datapoints_openpose_and_mediapipe(csv_path, frame_path)
    if mediapipe_data_extraction:
        collect_signing_right_left_hand_datapoints_from_mediapipe(csv_path, frame_path)
    if openpose_data_extraction:
        collect_signing_right_left_hand_datapoints_from_openpose(csv_path, frame_path)








'+++++++++++++++++++++++++++++'


def extract_mediapipe_keypoints_from_JSON(json_file):
    all_hands = []
    with open(json_file, "r") as signer_file:
        signer = json.load(signer_file)
    poselmList = signer["body_pose_keypoints"]
    leftHandlmList = signer["left_hand"]
    rightHandlmList = signer["right_hand"]

    flat_right_hand_list = [item for sublist in rightHandlmList for item in sublist]
    flat_left_hand_list = [item for sublist in leftHandlmList for item in sublist]

    if sum(flat_right_hand_list) == 0:
        all_hands.append(None)
    else:
        all_hands.append(create_hand_dictionaries(rightHandlmList, "right"))

    if sum(flat_left_hand_list) == 0:
        all_hands.append(None)
    else:
        all_hands.append(create_hand_dictionaries(leftHandlmList, "left"))

    body_pos = extract_body_position(poselmList)
    # print(all_hands)
    return all_hands, body_pos, poselmList

def extract_body_position(landmark_list):
    xList = []
    yList = []
    for id, lm in enumerate(landmark_list):
        if lm[3] > 0.2:
            xList.append(lm[0])
            yList.append(lm[1])

    if len(xList) == 0 or len(yList) == 0:
        bbox = (0, 0, 0, 0)
    else:
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)

        boxW, boxH = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, boxW, boxH

    return bbox

def create_hand_dictionaries(landmark_list, type):
    hand = {}
    lmList = []
    xList = []
    yList = []
    for landmark in landmark_list:
        # print(landmark)
        landmark_without_vis = landmark[:-1]
        # print(landmark_without_vis)
        lmList.append(landmark_without_vis)
        if landmark_without_vis[0] >= 0:
            xList.append(landmark_without_vis[0])
        if landmark_without_vis[1] >= 0:
            yList.append(landmark_without_vis[1])

    xmin, xmax = min(xList), max(xList)
    ymin, ymax = min(yList), max(yList)

    boxW, boxH = xmax - xmin, ymax - ymin
    bbox = xmin, ymin, boxW, boxH
    center_x, center_y = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
    hand["lmList"] = lmList
    hand["bbox"] = bbox
    hand["center"] = (center_x, center_y)
    hand["type"] = type
    hand["nosePos"] = (0, 0, 0, 0)
    hand["noseDistance"] = 0
    hand["signing"] = 0

    return hand