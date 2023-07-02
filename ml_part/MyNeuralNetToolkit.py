import os
import torch
import torch.nn as nn
import torch.nn.functional as activation_function
import numpy
import numpy as np
import pandas as pd
import pickle
import torch.utils.data as data
import torch.optim as optim
from beam_search import beam_search
from language_model import LanguageModel
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image
import time
"+++++++++++++++++++++++CER WER AND TEXT TRANSFORM+++++++++++++++++++++++"


def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):

    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    if ignore_case is True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)

    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''
    #print(reference)
    reference = join_char.join(filter(None, reference.split(' ')))
    #print(reference)
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)
    #print("WER edit distance",edit_distance)
    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")
    #print(f"WER edit_distance: {edit_distance} and len : {ref_len}")
    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,remove_space)
    #print("CER edit distance",edit_distance)
    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return edit_distance, ref_len, cer


class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        <SPACE> 0
        a 1
        b 2
        c 3
        d 4
        e 5
        f 6
        g 7
        h 8
        i 9
        j 10
        k 11
        l 12
        m 13
        n 14
        o 15
        p 16
        q 17
        r 18
        s 19
        t 20
        u 21
        v 22
        w 23
        x 24
        y 25
        z 26
        """

        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[0] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]

            if ch == numpy.nan:
                print("Ch is numpy nun")
            if ch == numpy.inf:
                print("Ch is numpy inf")

            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])

        return ''.join(string).replace('<SPACE>', ' ')


def create_directories(lexicon):
    parent = r"C:\Users\Alekos\Desktop\Διπλωματική\hand_classification"
    os.chdir(parent)
    mode = 0o666
    for i in lexicon:
        os.mkdir(i,mode)



"+++++++++++++++++++++++DATA EXTRACTION AND PREPROCCESSING+++++++++++++++++++++++"


def data_extraction(path, set_partition, offset, file_name, my_lexicon_file):
    csv_path = r"D:\University\ChicagoFSWild\ChicagoFSWild\ChicagoFSWild.csv"
    print(f"I do {path} now")
    skip_characters = r"@&'."
    if os.path.isfile(file_name):
        open_file = open(file_name, "rb")
        data = pickle.load(open_file)
        open_file.close()
    else:
        df = pd.read_csv(csv_path, usecols=['filename', 'number_of_frames', 'label_proc', 'partition'])

        set = numpy.load(path, allow_pickle=True)
        print(set.shape)
        set = set.tolist()
        data = []

        for idx in df.index:
            if df['partition'][idx] == set_partition:
                frame_num = df["number_of_frames"][idx]#when processing the first train_set.npy, val_set.npy, test_set.npy

                set[idx-offset] = set[idx-offset][0:frame_num]

                name = df["filename"][idx]
                sequence = df["label_proc"][idx]

                #my_lexicon_file.write(sequence)
                #my_lexicon_file.write(" ")

                coordinates = set[idx-offset]

                #coordinates = coordinates.tolist()


                if any(elem in sequence for elem in skip_characters) or name == 'youtube_4/sean_berdy_5756' or name == 'youtube_5/jeffrey_spinale_6043':
                #if name == 'youtube_4/sean_berdy_5756' or name == 'youtube_5/jeffrey_spinale_6043' or name == 'aslthat/joseph_wheeler_0472' or name == 'aslized/suzanne_stecker_0278' or name == 'awti/austin_andrews_0695' or name == 'deafvideo_2/frekky_1425' or name == 'deafvideo_2/confederateboy_1664':
                    print("I have to skip this")
                else:

                    for idx1, coordinate in enumerate(coordinates):
                        #coordinates[idx1] = rescale_3d_dim(coordinates[idx1])
                        coordinates[idx1] = wrist_normalization(coordinates[idx1])
                        #coordinates[idx1] = discard_3d_dim(coordinates[idx1])

                    # max_value = max([max(index) for index in zip(*coordinates)])
                    # min_value = min([min(index) for index in zip(*coordinates)])
                    #
                    # for idx2, coordinate in enumerate(coordinates):
                    #     coordinates[idx1] = data_normalization(coordinates[idx2], min_value, max_value)

                    #print(numpy.array(coordinates).shape)
                    sc = hand_size_normalization(coordinates)
                    if sc == 0:
                        pass
                        #print("Scale value equals zero")
                    else:
                        for idx2, coordinate in enumerate(coordinates):
                            coordinates[idx2] = [keypoint/sc for keypoint in coordinate]


                    data.append((numpy.array(coordinates),sequence))

        open_file = open(file_name, "wb")
        pickle.dump(data, open_file)
        open_file.close()


    return data


def data_extraction_two_hands(right_hand_path, left_hand_path, set_partition, offset, file_name, my_lexicon_file):
    csv_path = r"D:\University\ChicagoFSWild\ChicagoFSWild\ChicagoFSWild.csv"
    print(f"I do {right_hand_path} and  {left_hand_path} now")
    skip_characters = r"@&'."
    if os.path.isfile(file_name):
        open_file = open(file_name, "rb")
        data = pickle.load(open_file)
        open_file.close()
    else:
        df = pd.read_csv(csv_path, usecols=['filename', 'number_of_frames', 'label_proc', 'partition'])

        right_hand_set = numpy.load(right_hand_path, allow_pickle = True)
        left_hand_set = numpy.load(left_hand_path, allow_pickle = True)
        #print(righ.shape)
        right_hand_set = right_hand_set.tolist()
        left_hand_set = left_hand_set.tolist()
        data = []

        for idx in df.index:
            if df['partition'][idx] == set_partition:
                frame_num = df["number_of_frames"][idx]
                # print(type(frame_num))

                name = df["filename"][idx]
                sequence = df["label_proc"][idx]

                # my_lexicon_file.write(sequence)
                # my_lexicon_file.write(" ")

                right_coordinates = right_hand_set[idx - offset].tolist()
                left_coordinates = left_hand_set[idx-offset].tolist()

                angle = random.randint(0, 90)
                rotated_coordinates = []
                # rotated_coordinates_45 = []
                if any(elem in sequence for elem in skip_characters) or name == 'youtube_4/sean_berdy_5756' or name == 'youtube_5/jeffrey_spinale_6043':
                    # if name == 'youtube_4/sean_berdy_5756' or name == 'youtube_5/jeffrey_spinale_6043' or name == 'aslthat/joseph_wheeler_0472' or name == 'aslized/suzanne_stecker_0278' or name == 'awti/austin_andrews_0695' or name == 'deafvideo_2/frekky_1425' or name == 'deafvideo_2/confederateboy_1664':
                    print("I have to skip this")
                else:


                    for idx1, right_coordinate in enumerate(right_coordinates):
                        right_coordinates[idx1] = wrist_normalization(right_coordinates[idx1])
                        #right_coordinates[idx1] = discard_3d_dim(right_coordinates[idx1])


                    for idx2, left_coordinate in enumerate(left_coordinates):
                        left_coordinates[idx2] = wrist_normalization(left_coordinates[idx2])
                        #left_coordinates[idx2] = discard_3d_dim(left_coordinates[idx2])

                    sc_right = hand_size_normalization(right_coordinates)
                    if sc_right == 0:
                        print("Scale value equals zero")
                    else:
                        for idx3, r_coordinate in enumerate(right_coordinates):
                            right_coordinates[idx3] = [keypoint / sc_right for keypoint in r_coordinate]

                    sc_left = hand_size_normalization(left_coordinates)
                    if sc_left == 0:
                        print("Scale value equals zero")
                    else:
                        for idx4, l_coordinate in enumerate(left_coordinates):
                            left_coordinates[idx4] = [keypoint / sc_left for keypoint in l_coordinate]
                    # max_value_right = max([max(index) for index in zip(*right_coordinates)])
                    # min_value_right = min([min(index) for index in zip(*right_coordinates)])
                    # max_value_left = max([max(index) for index in zip(*left_coordinates)])
                    # min_value_left = min([min(index) for index in zip(*left_coordinates)])
                    #
                    # for idx3, right_coordinate in enumerate(right_coordinates):
                    #     right_coordinates[idx3] = data_normalization(right_coordinates[idx3], min_value_right, max_value_right)
                    #
                    #
                    # for idx4, left_coordinate in enumerate(left_coordinates):
                    #     left_coordinates[idx4] = data_normalization(left_coordinates[idx4], min_value_left, max_value_left)



                    data.append((numpy.array(right_coordinates), numpy.array(left_coordinates), sequence))

        open_file = open(file_name, "wb")
        pickle.dump(data, open_file)
        open_file.close()

    return data


def data_extraction_all_body_parts(right_hand_path, left_hand_path, body_path, set_partition, offset, file_name, my_lexicon_file):
    csv_path = r"D:\University\ChicagoFSWild\ChicagoFSWild\ChicagoFSWild.csv"
    print(f"I do {right_hand_path} and  {left_hand_path} and {body_path} now")
    skip_characters = r"@&'."
    if os.path.isfile(file_name):
        open_file = open(file_name, "rb")
        data = pickle.load(open_file)
        open_file.close()
    else:
        df = pd.read_csv(csv_path, usecols=['filename', 'number_of_frames', 'label_proc', 'partition'])

        right_hand_set = numpy.load(right_hand_path, allow_pickle = True)
        left_hand_set = numpy.load(left_hand_path, allow_pickle = True)
        body_parts_set = numpy.load(body_path, allow_pickle = True)
        #print(righ.shape)
        right_hand_set = right_hand_set.tolist()
        left_hand_set = left_hand_set.tolist()
        body_parts_set = body_parts_set.tolist()
        data = []

        for idx in df.index:
            if df['partition'][idx] == set_partition:
                frame_num = df["number_of_frames"][idx]
                # print(type(frame_num))

                name = df["filename"][idx]
                sequence = df["label_proc"][idx]

                # my_lexicon_file.write(sequence)
                # my_lexicon_file.write(" ")

                right_coordinates = right_hand_set[idx - offset]
                left_coordinates = left_hand_set[idx-offset]
                body_coordinates = body_parts_set[idx-offset]
                body_coordinates_without_face = []

                angle = random.randint(0, 90)
                rotated_coordinates = []
                # rotated_coordinates_45 = []
                if any(elem in sequence for elem in skip_characters) or name == 'youtube_4/sean_berdy_5756' or name == 'youtube_5/jeffrey_spinale_6043':
                    # if name == 'youtube_4/sean_berdy_5756' or name == 'youtube_5/jeffrey_spinale_6043' or name == 'aslthat/joseph_wheeler_0472' or name == 'aslized/suzanne_stecker_0278' or name == 'awti/austin_andrews_0695' or name == 'deafvideo_2/frekky_1425' or name == 'deafvideo_2/confederateboy_1664':
                    print("I have to skip this")
                else:
                    for idx1, right_coordinate in enumerate(right_coordinates):
                        right_coordinates[idx1] = wrist_normalization(right_coordinates[idx1])
                        #print(len(right_coordinates[idx]))
                        #right_coordinates[idx] = discard_3d_dim(right_coordinates[idx])

                    for idx2, left_coordinate in enumerate(left_coordinates):
                        left_coordinates[idx2] = wrist_normalization(left_coordinates[idx2])
                        #left_coordinates[idx] = discard_3d_dim(left_coordinates[idx])

                    for idx3, body_coordinate in enumerate(body_coordinates):
                        #print("Before",body_coordinates[idx3])
                        #print(len(body_coordinate))
                        body_coordinates_without_face.append(body_coordinates[idx3][33:])
                        #print("After", body_coordinates_without_face[idx3])
                        #print(len(body_coordinates_without_face[idx3]))
                    max_value = max([max(index) for index in zip(*right_coordinates)])
                    min_value = min([min(index) for index in zip(*right_coordinates)])
                    # for idx,coordinate in enumerate(coordinates):
                    # if set_partition != 'test':
                    #   rotated_coordinates.append(data_normalization(coordinate_rotation(coordinates[idx],math.radians(angle)),min_value,max_value))
                    # rotated_coordinates_45.append(data_normalization(coordinate_rotation(coordinates[idx],math.radians(45)),min_value,max_value))
                    #    coordinates[idx] = data_normalization(coordinates[idx],min_value,max_value)
                    # if set_partition != 'test':
                    #    data.append((numpy.array(rotated_coordinates),sequence))
                    # data.append((numpy.array(rotated_coordinates_45), sequence))
                    # print(numpy.array(coordinates).shape)
                    data.append((numpy.array(right_coordinates), numpy.array(left_coordinates), numpy.array(body_coordinates_without_face), sequence))

        open_file = open(file_name, "wb")
        pickle.dump(data, open_file)
        open_file.close()

    return data


def prepare_data_for_stt(path, set_partition, offset):
    csv_path = r"D:\University\ChicagoFSWild\ChicagoFSWild\ChicagoFSWild.csv"

    skip_characters = r"@&'."
    df = pd.read_csv(csv_path, usecols=['filename', 'number_of_frames', 'label_proc', 'partition'])

    set = numpy.load(path)
    print(set.shape)
    set = set.tolist()
    data = []
    dfObj = pd.DataFrame(columns=['filename', 'filesize', 'transcript'])
    for idx in df.index:
        if df['partition'][idx] == set_partition:
            frame_num = df["number_of_frames"][idx]
            # print(type(frame_num))
            set[idx - offset] = set[idx - offset][0:frame_num]
            name = df["filename"][idx] + ".npy"
            name = name.replace("/", "_")
            sequence = df["label_proc"][idx]
            dfObj = dfObj.append({'filename': name, 'filesize': frame_num, 'transcript': sequence}, ignore_index=True)

            coordinates = set[idx - offset]

            if any(elem in sequence for elem in skip_characters) or name == 'youtube_4/sean_berdy_5756' or name == 'youtube_5/jeffrey_spinale_6043':
                # if name == 'youtube_4/sean_berdy_5756' or name == 'youtube_5/jeffrey_spinale_6043' or name == 'aslthat/joseph_wheeler_0472' or name == 'aslized/suzanne_stecker_0278' or name == 'awti/austin_andrews_0695' or name == 'deafvideo_2/frekky_1425' or name == 'deafvideo_2/confederateboy_1664':
                print("I have to skip this")
            else:
                os.chdir(r"C:\Users\Alekos\PycharmProjects\pytorch_tutorials\hand_coordinates")
                if os.path.isfile(name) == False:
                    with open(name, 'wb') as f:
                        numpy.save(f, numpy.array(coordinates))

    if set_partition == "train":
        csv_file = "train.csv"
    if set_partition == "dev":
        csv_file = "dev.csv"
    if set_partition == "test":
        csv_file = "test.csv"
    os.chdir(r"C:\Users\Alekos\PycharmProjects\pytorch_tutorials")
    if os.path.isfile(csv_file) == False:
        dfObj.to_csv(csv_file)


def discard_3d_dim(coordinates):
     reduced_coordinate_list = []
     iter = 2
     for idx, ele in enumerate(coordinates):
         if idx != iter:
             reduced_coordinate_list.append(ele)
         else:
             iter = iter + 3

     return reduced_coordinate_list


def min_max_scaling_normalization(coordinates, min_ele, max_ele):

    non_zeros = np.count_nonzero(coordinates)
    if non_zeros != 0:
        for idx, element in enumerate(coordinates):
            coordinates[idx] = (element - min_ele)/(max_ele - min_ele)

    return coordinates


def wrist_normalization(coordinate_list):
    #print(type(coordinate_list))
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
    #print(type(new_list))
    return new_list


def hand_size_normalization(hand_list):

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

        distance1 = np.sqrt(np.sum(np.square(np.array(restructured_coord_list[5]) - np.array(restructured_coord_list[0]))))
        distance2 = np.sqrt(np.sum(np.square(np.array(restructured_coord_list[9]) - np.array(restructured_coord_list[0]))))
        distance3 = np.sqrt(np.sum(np.square(np.array(restructured_coord_list[13]) - np.array(restructured_coord_list[0]))))
        distance4 = np.sqrt(np.sum(np.square(np.array(restructured_coord_list[17]) - np.array(restructured_coord_list[0]))))
        total_distance = total_distance + distance1 + distance2 + distance3 + distance4

    scale_value = total_distance/(4*len(hand_list))

    return scale_value


def rescale_3d_dim(coordinates):
    rescaled_coordinate_list = []
    iter = 2
    for idx, ele in enumerate(coordinates):
        if idx != iter:
            rescaled_coordinate_list.append(ele)
        else:
            ele = ele/100
            rescaled_coordinate_list.append(ele)
            iter = iter + 3

    return rescaled_coordinate_list



def train_test_split():
    csv_path = r"D:\University\ChicagoFSWild\ChicagoFSWild\ChicagoFSWild.csv"
    skip_characters = r"@&'."

    df = pd.read_csv(csv_path, usecols=['filename', 'number_of_frames', 'label_proc', 'partition'])
    train_videos = []
    train_videos_labels = []
    val_videos = []
    val_videos_labels = []
    test_videos = []
    test_videos_labels = []

    for idx in df.index:
        video_name = df["filename"][idx]
        partition = df["partition"][idx]
        sequence = df["label_proc"][idx]

        if any(elem in sequence for elem in skip_characters) or video_name == 'youtube_4/sean_berdy_5756' or video_name == 'youtube_5/jeffrey_spinale_6043':
            # if name == 'youtube_4/sean_berdy_5756' or name == 'youtube_5/jeffrey_spinale_6043' or name == 'aslthat/joseph_wheeler_0472' or name == 'aslized/suzanne_stecker_0278' or name == 'awti/austin_andrews_0695' or name == 'deafvideo_2/frekky_1425' or name == 'deafvideo_2/confederateboy_1664':
            print("I have to skip this")
        else:
            if partition == "train":
                train_videos.append(video_name)
                train_videos_labels.append(sequence)
            if partition == "dev":
                val_videos.append(video_name)
                val_videos_labels.append(sequence)
            if partition == "test":
                test_videos.append(video_name)
                test_videos_labels.append(sequence)

    return train_videos,train_videos_labels,val_videos,val_videos_labels,test_videos,test_videos_labels


def data_processing(data):
    text_transform = TextTransform()
    hand_sequences = []
    labels = []
    #filenames = []
    input_lengths = []
    label_lengths = []
    for id, (landmarks, sentence) in enumerate(data):
        landmarks = torch.from_numpy(landmarks).float()
        hand_sequences.append(landmarks)

        label = torch.Tensor(text_transform.text_to_int(sentence.lower()))

        labels.append(label)
        #filenames.append(filename)
        input_lengths.append(landmarks.shape[0])
        label_lengths.append(len(label))


    hand_sequences = nn.utils.rnn.pad_sequence(hand_sequences, batch_first=True).unsqueeze(1).transpose(2, 3)
    #print(hand_sequences.shape)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    #print(labels.shape)
    return hand_sequences, labels, input_lengths, label_lengths


def data_processing_2_hands(data):
    text_transform = TextTransform()
    right_hand_sequences = []
    left_hand_sequences = []
    labels = []
    # filenames = []
    input_lengths = []
    label_lengths = []
    for id, (right_hand_landmarks, left_hand_landmarks, sentence) in enumerate(data):
        right_hand_landmarks = torch.from_numpy(right_hand_landmarks).float()
        left_hand_landmarks = torch.from_numpy(left_hand_landmarks).float()
        right_hand_sequences.append(right_hand_landmarks)
        left_hand_sequences.append(left_hand_landmarks)

        label = torch.Tensor(text_transform.text_to_int(sentence.lower()))

        labels.append(label)
        # filenames.append(filename)
        input_lengths.append(right_hand_landmarks.shape[0])
        label_lengths.append(len(label))

    right_hand_sequences = nn.utils.rnn.pad_sequence(right_hand_sequences, batch_first=True).unsqueeze(1).transpose(2, 3)
    left_hand_sequences = nn.utils.rnn.pad_sequence(left_hand_sequences, batch_first=True).unsqueeze(1).transpose(2, 3)
    # print(hand_sequences.shape)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    # print(labels.shape)
    return right_hand_sequences, left_hand_sequences, labels, input_lengths, label_lengths


def data_processing_all_body_parts(data):
    text_transform = TextTransform()
    right_hand_sequences = []
    left_hand_sequences = []
    body_pose_sequences = []
    labels = []
    # filenames = []
    input_lengths = []
    label_lengths = []
    for id, (right_hand_landmarks, left_hand_landmarks, body_pose_landmarks, sentence) in enumerate(data):
        right_hand_landmarks = torch.from_numpy(right_hand_landmarks).float()
        left_hand_landmarks = torch.from_numpy(left_hand_landmarks).float()
        body_pose_landmarks = torch.from_numpy(body_pose_landmarks).float()
        right_hand_sequences.append(right_hand_landmarks)
        left_hand_sequences.append(left_hand_landmarks)
        body_pose_sequences.append(body_pose_landmarks)

        label = torch.Tensor(text_transform.text_to_int(sentence.lower()))

        labels.append(label)
        # filenames.append(filename)
        input_lengths.append(right_hand_landmarks.shape[0])
        label_lengths.append(len(label))

    right_hand_sequences = nn.utils.rnn.pad_sequence(right_hand_sequences, batch_first=True).unsqueeze(1).transpose(2,3)
    left_hand_sequences = nn.utils.rnn.pad_sequence(left_hand_sequences, batch_first=True).unsqueeze(1).transpose(2, 3)
    body_pose_sequences = nn.utils.rnn.pad_sequence(body_pose_sequences, batch_first=True).unsqueeze(1).transpose(2, 3)
    # print(hand_sequences.shape)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    # print(labels.shape)
    return right_hand_sequences, left_hand_sequences, body_pose_sequences, labels, input_lengths, label_lengths


def image_processing(data):
    text_transform = TextTransform()
    image_sequences = []
    labels = []
    input_lengths = []
    label_lengths = []
    for id, (images, sentence) in enumerate(data):
        image_sequences.append(images)

        label = torch.Tensor(text_transform.text_to_int(sentence.lower()))

        labels.append(label)
        input_lengths.append(images.shape[0])
        label_lengths.append(len(label))


    image_sequences = nn.utils.rnn.pad_sequence(image_sequences, batch_first=True)

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return image_sequences, labels, input_lengths, label_lengths


def image_processing_temp(data):
    text_transform = TextTransform()
    image_sequences = []
    labels = []
    input_lengths = []
    label_lengths = []
    for id, (images, sentence) in enumerate(data):
        image_sequences.append(images)

        label = torch.Tensor(text_transform.text_to_int(sentence.lower()))

        labels.append(label)
        input_lengths.append(images.shape[0]//2)
        label_lengths.append(len(label))

    #print(image_sequences)
    image_sequences = nn.utils.rnn.pad_sequence(image_sequences, batch_first=True).unsqueeze(1).transpose(2, 3)

    #.unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return image_sequences, labels, input_lengths, label_lengths


"+++++++++++++++++++++++DECODERS+++++++++++++++++++++++"

def already_implemnted_beam_search(output, lm):
    chars = " abcdefghijklmnopqrstuvwxyz"
    output_batch = output.tolist()
    decodes = []
    for output in output_batch:
        mat = numpy.array(output)
        res = beam_search(mat, chars, lm=lm)
        decodes.append(res)

    return decodes



def BeamSearchDecoder(output, labels, label_lengths, k, blank_label = 27, collapse_repeated=True):

    #print(output.shape)
    output_batch = output.tolist()
    decoded_outputs = []
    for output in output_batch:
        output_sequences = [[list(), 1.0]]
        # walk over each step in sequence
        #print(len(output))
        for token_probs in output:
            new_sequences = []
            # expand each current candidate
            for i in range(len(output_sequences)):
                seq, score = output_sequences[i]
                for j in range(len(token_probs)):
                    new_seq = seq + [j]
                    new_score = score * abs(token_probs[j])
                    new_sequences.append((new_seq, new_score))
            # order all candidates by score
            output_sequences = sorted(new_sequences, key=lambda val: val[1], reverse=False)

            # select k best
            output_sequences = output_sequences[:k]
        #print(output_sequences)
        decoded_outputs.append(torch.Tensor(output_sequences[0][0]))
    #print(decoded_outputs)
    decoded_outputs = nn.utils.rnn.pad_sequence(decoded_outputs, batch_first=True)
    arg_maxes = torch.as_tensor(decoded_outputs)
    decodes = []
    targets = []

    text_transform = TextTransform()
    # print("labels",labels)
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())

        decodes.append(text_transform.int_to_text(decode))
    # print("Decodes", decodes)
    # print("Targets",targets)
    return decodes, targets


def GreedyDecoder(output, labels, label_lengths, blank_label=27, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    text_transform = TextTransform()

    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())

        decodes.append(text_transform.int_to_text(decode))

    return decodes, targets


"+++++++++++++++++++++++NEURAL NETWORKS+++++++++++++++++++++++"

class bidirectional_lstm(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(bidirectional_lstm, self).__init__()

        self.BiLSTM = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.layer_norm(x)
        x = activation_function.gelu(x)

        x, _ = self.BiLSTM(x)
        x = self.dropout(x)

        return x


class bidirectional_gru(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(bidirectional_gru, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.layer_norm(x)

        x = activation_function.gelu(x)

        x, _ = self.BiGRU(x)

        x = self.dropout(x)


        return x


class speech_recognition_model_brnn_signing_hand(nn.Module):

    def __init__(self, n_rnn_layers, n_class, n_feats, dropout=0.1):
        super(speech_recognition_model_brnn_signing_hand, self).__init__()

        #print(n_feats)
        #self.embedding = nn.Embedding(n_class,n_feats)
        self.fully_connected = nn.Linear(n_feats, n_feats*2)
        #self.birnn_layers = BidirectionalLSTM(rnn_dim=n_feats*2,hidden_size=n_feats*2, dropout=dropout, batch_first=True)
        self.birnn_layers = nn.Sequential(*[bidirectional_gru(rnn_dim=2 * n_feats if i == 0 else n_feats * 4,
                                                              hidden_size=n_feats*2, dropout=dropout, batch_first=i == 0)
                                            for i in range(n_rnn_layers)
                                            ])
        self.classifier = nn.Sequential(
            nn.Linear(n_feats*4, n_feats*2),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_feats*2, n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_feats, n_class)
        )

    def forward(self, x):
        sizes = x.size()
        #x = x.float32()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        #x = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True,enforce_sorted=False)
        x = self.birnn_layers(x)
        # x,_ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.classifier(x)
        return x


class speech_recognition_model_brnn_two_hands(nn.Module):

    def __init__(self, n_rnn_layers, n_class, n_feats, dropout=0.1):
        super(speech_recognition_model_brnn_two_hands, self).__init__()

        #print(n_feats)
        #self.embedding = nn.Embedding(n_class,n_feats)
        self.fully_connected_right = nn.Linear(n_feats, n_feats*2)
        self.fully_connected_left = nn.Linear(n_feats, n_feats*2)
        #self.birnn_layers = BidirectionalLSTM(rnn_dim=n_feats*2,hidden_size=n_feats*2, dropout=dropout, batch_first=True)
        self.birnn_layers = nn.Sequential(*[bidirectional_lstm(rnn_dim=4 * n_feats if i == 0 else n_feats * 8,
                                                               hidden_size=4*n_feats, dropout=dropout, batch_first=i == 0)
                                            for i in range(n_rnn_layers)
                                            ])
        self.classifier = nn.Sequential(
            nn.Linear(n_feats*8, n_feats * 4),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_feats * 4, n_feats * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_feats * 2, n_class)
        )

    def forward(self, x_right, x_left):
        right_sizes = x_right.size()
        left_sizes = x_left.size()
        #x = x.float32()
        x_right = x_right.view(right_sizes[0], right_sizes[1] * right_sizes[2], right_sizes[3])  # (batch, feature, time)
        x_left = x_left.view(left_sizes[0], left_sizes[1] * left_sizes[2], left_sizes[3])
        x_right = x_right.transpose(1, 2)  # (batch, time, feature)
        x_left = x_left.transpose(1, 2)
        x_right = self.fully_connected_right(x_right)
        x_left = self.fully_connected_left(x_left)
        both_hands = torch.cat((x_right, x_left),2)
        rnn_out = self.birnn_layers(both_hands)
        out = self.classifier(rnn_out)
        return out


class speech_recognition_model_brnn_two_hands_v2(nn.Module):
    def __init__(self, n_rnn_layers, n_class, n_feats, dropout=0.1):
        super(speech_recognition_model_brnn_two_hands_v2, self).__init__()

        #print(n_feats)
        #self.embedding = nn.Embedding(n_class,n_feats)

        self.fully_connected = nn.Linear(n_feats, n_feats*2)
        #self.birnn_layers = BidirectionalLSTM(rnn_dim=n_feats*2,hidden_size=n_feats*2, dropout=dropout, batch_first=True)
        self.birnn_layers = nn.Sequential(*[bidirectional_gru(rnn_dim=2 * n_feats if i == 0 else n_feats * 4,
                                                              hidden_size=2*n_feats, dropout=dropout, batch_first=i == 0)
                                            for i in range(n_rnn_layers)
                                            ])
        self.classifier = nn.Sequential(
            nn.Linear(n_feats*4, n_feats*2),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_feats*2,n_class),
        )

    def forward(self, x_right, x_left):
        right_sizes = x_right.size()
        left_sizes = x_left.size()
        #x = x.float32()
        x_right = x_right.view(right_sizes[0], right_sizes[1] * right_sizes[2], right_sizes[3])  # (batch, feature, time)
        x_left = x_left.view(left_sizes[0], left_sizes[1] * left_sizes[2], left_sizes[3])
        x_right = x_right.transpose(1, 2)  # (batch, time, feature)
        x_left = x_left.transpose(1, 2)
        both_hands = torch.cat((x_right, x_left), 2)
        both_hands = self.fully_connected(both_hands)
        rnn_out = self.birnn_layers(both_hands)
        out = self.classifier(rnn_out)
        return out


class speech_recognition_model_brnn_all_body_parts(nn.Module):

    def __init__(self, n_rnn_layers, n_class, n_feats, body_part_feats, dropout=0.1):
        super(speech_recognition_model_brnn_all_body_parts, self).__init__()

        #print(n_feats)
        #self.embedding = nn.Embedding(n_class,n_feats)
        self.fully_connected_right = nn.Linear(n_feats, n_feats*2)
        self.fully_connected_left = nn.Linear(n_feats, n_feats*2)
        self.fully_connected_body = nn.Linear(body_part_feats, body_part_feats*2)
        #self.birnn_layers = BidirectionalLSTM(rnn_dim=n_feats*2,hidden_size=n_feats*2, dropout=dropout, batch_first=True)
        self.birnn_layers = nn.Sequential(*[bidirectional_gru(rnn_dim=(4 * n_feats + body_part_feats * 2) if i == 0 else (4 * n_feats + body_part_feats * 2) * 2,
                                                              hidden_size=(4*n_feats+body_part_feats*2), dropout=dropout, batch_first=i == 0)
                                            for i in range(n_rnn_layers)
                                            ])
        self.classifier = nn.Sequential(
            nn.Linear((4*n_feats+body_part_feats*2) * 2, (4*n_feats+body_part_feats*2)),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear((4*n_feats+body_part_feats*2),n_class),
        )

    def forward(self, x_right, x_left, x_body):
        right_sizes = x_right.size()
        left_sizes = x_left.size()
        body_sizes = x_body.size()

        x_right = x_right.view(right_sizes[0], right_sizes[1] * right_sizes[2], right_sizes[3])  # (batch, feature, time)
        x_left = x_left.view(left_sizes[0], left_sizes[1] * left_sizes[2], left_sizes[3])
        x_body = x_body.view(body_sizes[0], body_sizes[1]*body_sizes[2], body_sizes[3])

        x_right = x_right.transpose(1, 2)  # (batch, time, feature)
        x_left = x_left.transpose(1, 2)
        x_body = x_body.transpose(1, 2)

        x_right = self.fully_connected_right(x_right)
        x_left = self.fully_connected_left(x_left)
        x_body = self.fully_connected_body(x_body)

        all_parts = torch.cat((x_right, x_left, x_body),2)

        rnn_out = self.birnn_layers(all_parts)

        out = self.classifier(rnn_out)

        return out


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


"+++++++++++++++++++++++TRAIN TESTING AND NN INITIALIZATION+++++++++++++++++++++++"


def BRNN_signing_train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter):
    model.train()
    data_len = len(train_loader.dataset)
    train_loss = 0
    for batch_idx, _data in enumerate(train_loader):
        #print(batch_idx)
        hand_coords, labels, input_lengths, label_lengths = _data
        hand_coords, labels = hand_coords.to(device), labels.to(device)


        optimizer.zero_grad()
        output = model(hand_coords)  # (batch, time, n_class)
        #make_dot(output.mean(), params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
        output = nn.functional.log_softmax(output, dim=2)

        output = output.transpose(0,1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths,label_lengths)

        #print(output.shape)
        if torch.isinf(loss):
            print(f"Loss is inf in {batch_idx}")
            print(output.shape)
            print(labels.shape)
            #print(output)
        if torch.isnan(loss):
            print(f"Loss is nan in {batch_idx}")
            print(output.shape)
            #print(output)
            print(labels.shape)

        if torch.isinf(loss) or torch.isnan(loss):
            pass
        else:
            train_loss += loss.item() / len(train_loader)
            loss.backward()



        optimizer.step()
        scheduler.step()
        iter_meter.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(hand_coords), data_len,
                       100. * batch_idx / len(train_loader), loss.item()))
        #if batch_idx == data_len:
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss,model


def BRNN_signing_eval(model, device, test_loader, criterion, beam_search):
    print('\nevaluating...')
    data_len = len(test_loader.dataset)
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    test_cer_beam, test_wer_beam = [],[]
    avg_cer_beam = 0
    avg_wer_beam = 0
    total_edit_distance = 0
    total_ref_lens = 0
    test_cer_2, test_wer_2 = [], []
    chars = " abcdefghijklmnopqrstuvwxyz"
    with open("my_lexicon.txt") as f:
         contents = f.readlines()
    lm1 = LanguageModel(contents[0], chars)

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            hand_coords, labels, input_lengths, label_lengths= _data
            hand_coords, labels = hand_coords.to(device), labels.to(device)

            output = model(hand_coords)  # (batch, time, n_class)
            if beam_search:
                decoded_preds_beam = already_implemnted_beam_search(nn.functional.softmax(output, dim=2),lm = None)
            output = nn.functional.log_softmax(output, dim=2)
            output = output.transpose(0,1) # (time, batch, n_class)



            loss = criterion(output, labels, input_lengths, label_lengths)
            if torch.isinf(loss) or torch.isnan(loss):
                print(hand_coords)
                print(output.shape)
                print(output)
                print(labels.shape)
                print(labels)
                print("Inf loss")
                pass
            else:
                test_loss += loss.item() / len(test_loader)


            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            # if beam_search:
            #     decoded_preds_beam_lm = already_implemnted_beam_search(output.transpose(0, 1),lm)
            #decoded_preds_beam, decoded_targets_beam = BeamSearchDecoder(output.transpose(0, 1), labels, label_lengths,1)
            #print(filenames)
            #classify_hand_images(output.transpose(0, 1), r"C:\Users\Alekos\Desktop\Διπλωματική\MediaPipe_hand_images", r"C:\Users\Alekos\Desktop\Διπλωματική\hand_classification",filenames[0],27)
            #print(f"Completion {i / len(test_loader)}")
            for j in range(len(decoded_preds)):



                # print("\n")
               # print("Beam serach",seq[j])
                dist,ref_len,cer1 = cer(decoded_targets[j], decoded_preds[j])
                total_edit_distance = total_edit_distance + dist
                total_ref_lens = total_ref_lens + ref_len
                test_cer.append(cer1)
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
                if beam_search:
                    distb,ref_lenb,beam_cer = cer(decoded_targets[j], decoded_preds_beam[j])
                    test_cer_beam.append(beam_cer)
                    test_wer_beam.append(wer(decoded_targets[j], decoded_preds_beam[j]))

                #test_cer_2.append(cer(decoded_targets_beam[j], decoded_preds_2[j]))
                #test_wer_2.append(wer(decoded_targets_beam[j], decoded_preds_2[j]))



    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    ger_cer = total_edit_distance / total_ref_lens
    if beam_search:
        avg_cer_beam = sum(test_cer_beam) / len(test_cer_beam)
        avg_wer_beam = sum(test_wer_beam) / len(test_wer_beam)

    #avg_cer_2 = sum(test_cer_2) / len(test_cer_2)
    #avg_wer_2 = sum(test_wer_2) / len(test_wer_2)
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f} and {}\n'.format(test_loss, avg_cer, avg_wer, ger_cer))
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f} for beam\n'.format(test_loss, avg_cer_beam, avg_wer_beam))
    #print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f} for beam\n'.format(test_loss,avg_cer_2,avg_wer_2))
    return test_loss,avg_cer,avg_wer


def BRNN_signing_hand_solution(train_dataset, val_dataset, test_dataset, learning_rate=5e-4, batch_size=20, epochs=50):
    hparams = {
        "n_rnn_layers": 2,
        "n_class": 28,
        "n_feats": 63,
        "dropout": 0.2,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    # experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Device {device}")
    if not os.path.isdir("./data"):
        os.makedirs("./data")


    #data_processing(train_dataset)
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=hparams['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing(x),
                                   **kwargs)
    val_loader = data.DataLoader(dataset=val_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(x),
                                  **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(x),
                                  **kwargs)

    model = speech_recognition_model_brnn_signing_hand(hparams['n_rnn_layers'], hparams['n_class'], hparams['n_feats'], hparams['dropout']).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CTCLoss(blank=27).to(device)
    #criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],steps_per_epoch=int(len(train_loader)),epochs=hparams['epochs'],anneal_strategy='linear')

    train_losses = []
    test_losses = []
    cers = []
    wers = []
    iter_meter = IterMeter()
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss,model = BRNN_signing_train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)
        val_loss,cer,wer = BRNN_signing_eval(model, device, val_loader, criterion, beam_search = False)
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        cers.append(cer)
        wers.append(wer)
    end = time.time()
    BRNN_signing_eval(model, device, test_loader, criterion, beam_search=True)
    save = 'n'
    if save == 'y':
        A = np.array(train_losses)
        B = np.array(test_losses)
        C = np.array(cers)
        D = np.array(wers)
        np.save(r'E:\My Models\signing_hand_3D\BLSTM_3D_coordinates_new\BLSTMx2_classifier2_dist_norm/BRNN_epoch_training_losses.npy', A)
        np.save(r'E:\My Models\signing_hand_3D\BLSTM_3D_coordinates_new\BLSTMx2_classifier2_dist_norm/BRNN_epoch_test_losses.npy', B)
        np.save(r'E:\My Models\signing_hand_3D\BLSTM_3D_coordinates_new\BLSTMx2_classifier2_dist_norm/cers.npy', C)
        np.save(r'E:\My Models\signing_hand_3D\BLSTM_3D_coordinates_new\BLSTMx2_classifier2_dist_norm/wers.npy', D)
        print(f"Total time: {(end - start) / 60:.3f} minutes")
        torch.save(model.state_dict(), r'E:\My Models\signing_hand_3D\BLSTM_3D_coordinates_new\BLSTMx2_classifier2_dist_norm\model_weights.pth')

        fig = plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(1, epochs + 1), A)  # train loss (on epoch end)
        plt.plot(np.arange(1, epochs + 1), B)  # test loss (on epoch end)
        plt.title("model loss")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc="upper left")
        # 2nd figure
        plt.subplot(122)
        plt.plot(np.arange(1, epochs + 1), C)  # CER accuracy (on epoch end)
        plt.plot(np.arange(1, epochs + 1), D)  # WER accuracy (on epoch end)
        plt.title("CER & WER")
        plt.xlabel('epochs')
        plt.ylabel('cer wer')
        plt.legend(['cer', 'wer'], loc="upper left")
        title = r"E:\My Models\signing_hand_3D\BLSTM_3D_coordinates_new\BLSTMx2_classifier2_dist_norm/cer_wer_figure.png"
        title_eps = r"E:\My Models\signing_hand_3D\BLSTM_3D_coordinates_new\BLSTMx2_classifier2_dist_norm/cer_wer_figure.eps"
        plt.savefig(title, dpi=900)
        plt.savefig(title_eps, dpi=900)
        # plt.close(fig)
        plt.show()

    #BRNN_eval(model, device, test_loader, criterion, beam_search = False)


def BRNN_signing_hand_testing(test_dataset, learning_rate=5e-4, batch_size=20):

    hparams = {
        "n_rnn_layers": 2,
        "n_class": 28,
        "n_feats": 63,
        "dropout": 0.2,
        "learning_rate": learning_rate,
        "batch_size": batch_size
    }

    # experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Device {device}")
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(x),
                                  **kwargs)

    model = speech_recognition_model_brnn_signing_hand(hparams['n_rnn_layers'], hparams['n_class'], hparams['n_feats'], hparams['dropout']).to(device)
    model.load_state_dict(torch.load('E:\My Models\signing_hand_3D\BGRU_3D_coordinates\BGRUx2_classifier3/model_weights.pth'))
    # model.config.ctc_zero_infinity = True
    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    criterion = nn.CTCLoss(blank=27).to(device)
    # criterion = nn.CrossEntropyLoss()


    iter_meter = IterMeter()
    #for epoch in range(1, epochs + 1):
    BRNN_signing_eval(model, device, test_loader, criterion, beam_search = True)

    #torch.save(model.state_dict(), 'model_weights.pth')


def BRNN_two_hand_train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter):
    model.train()
    data_len = len(train_loader.dataset)
    train_loss = 0
    for batch_idx, _data in enumerate(train_loader):
        hand_coords_right, hand_coords_left, labels, input_lengths, label_lengths = _data
        hand_coords_right, hand_coords_left, labels = hand_coords_right.to(device), hand_coords_left.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(hand_coords_right, hand_coords_left)  # (batch, time, n_class)
        output = nn.functional.log_softmax(output, dim=2)

        output = output.transpose(0,1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths,label_lengths)

        #print(output.shape)
        if torch.isinf(loss):
            print(f"Loss is inf in {batch_idx}")
            print(output.shape)
            print(labels.shape)
            #print(output)
        if torch.isnan(loss):
            print(f"Loss is nan in {batch_idx}")
            print(output.shape)
            #print(output)
            print(labels.shape)

        if torch.isinf(loss) or torch.isnan(loss):
            pass
        else:
            train_loss += loss.item() / len(train_loader)
            loss.backward()

        optimizer.step()
        scheduler.step()
        iter_meter.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(hand_coords_right), data_len,
                       100. * batch_idx / len(train_loader), loss.item()))
        #if batch_idx == data_len:
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss,model


def BRNN_two_hand_eval(model, device, test_loader, criterion, beam_search):
    print('\nevaluating...')
    data_len = len(test_loader.dataset)
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    test_cer_beam, test_wer_beam = [],[]
    avg_cer_beam = 0
    avg_wer_beam = 0
    chars = " abcdefghijklmnopqrstuvwxyz"
    with open("my_lexicon.txt") as f:
         contents = f.readlines()
    lm = LanguageModel(contents[0], chars)
    total_edit_distance = 0
    total_ref_lens = 0
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            hand_coords_right, hand_coords_left, labels, input_lengths, label_lengths = _data
            hand_coords_right, hand_coords_left, labels = hand_coords_right.to(device), hand_coords_left.to(device), labels.to(device)

            output = model(hand_coords_right, hand_coords_left)  # (batch, time, n_class)
            if beam_search:
                decoded_preds_beam = already_implemnted_beam_search(nn.functional.softmax(output, dim=2),lm = None)
            output = nn.functional.log_softmax(output, dim=2)
            output = output.transpose(0,1) # (time, batch, n_class)



            loss = criterion(output, labels, input_lengths, label_lengths)
            if torch.isinf(loss) or torch.isnan(loss):
                print(output.shape)
                print(labels.shape)
                print("Inf loss")
                pass
            else:
                test_loss += loss.item() / len(test_loader)


            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            #decoded_preds_beam_lm = already_implemnted_beam_search(output.transpose(0, 1),lm)
            #decoded_preds_beam, decoded_targets_beam = BeamSearchDecoder(output.transpose(0, 1), labels, label_lengths,1)
            #print(filenames)
            #classify_hand_images(output.transpose(0, 1), r"C:\Users\Alekos\Desktop\Διπλωματική\MediaPipe_hand_images", r"C:\Users\Alekos\Desktop\Διπλωματική\hand_classification",filenames[0],27)

            for j in range(len(decoded_preds)):
                #print("1->: ",decoded_preds_2[j])
                #print("2->: ",decoded_targets[j])

                # print("\n")
               # print("Beam serach",seq[j])
                edit_dst, ref_len, cer1 = cer(decoded_targets[j], decoded_preds[j])
                total_edit_distance = total_edit_distance + edit_dst
                total_ref_lens = total_ref_lens + ref_len
                test_cer.append(cer1)
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

                if beam_search:
                    beam_dist, ref_len, beam_cer = cer(decoded_targets[j], decoded_preds_beam[j])
                    test_cer_beam.append(beam_cer)
                    test_wer_beam.append(wer(decoded_targets[j], decoded_preds_beam[j]))

                #test_cer_2.append(cer(decoded_targets_beam[j], decoded_preds_2[j]))
                #test_wer_2.append(wer(decoded_targets_beam[j], decoded_preds_2[j]))


    ger_cer = total_edit_distance/total_ref_lens
    #print(total_edit_distance)
    #print(total_ref_lens)
    #print("CER",ger_cer)
    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    if beam_search:
        avg_cer_beam = sum(test_cer_beam) / len(test_cer_beam)
        avg_wer_beam = sum(test_wer_beam) / len(test_wer_beam)

    #avg_cer_2 = sum(test_cer_2) / len(test_cer_2)
    #avg_wer_2 = sum(test_wer_2) / len(test_wer_2)
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f} for beam\n'.format(test_loss, avg_cer_beam, avg_wer_beam))
    #print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f} for beam\n'.format(test_loss,avg_cer_2,avg_wer_2))
    return test_loss,avg_cer,avg_wer


def BRNN_two_hand_solution(train_dataset, val_dataset, test_dataset, learning_rate=5e-4, batch_size=20, epochs=50):
    hparams = {
        "n_rnn_layers": 2,
        "n_class": 28,
        "n_feats": 63,
        "dropout": 0.2,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    # experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Device {device}")
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    # data_processing(train_dataset)

    # print(train_data[0])
    # print(train_data[1])
    # data_processing(train_dataset)
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=hparams['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing_2_hands(x),
                                   **kwargs)
    val_loader = data.DataLoader(dataset=val_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing_2_hands(x),
                                  **kwargs)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing_2_hands(x),
                                  **kwargs)

    model = speech_recognition_model_brnn_two_hands(hparams['n_rnn_layers'], hparams['n_class'], hparams['n_feats'], hparams['dropout']).to(device)
    # model.config.ctc_zero_infinity = True
    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CTCLoss(blank=27).to(device)
    # criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                              steps_per_epoch=int(len(train_loader)), epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    train_losses = []
    test_losses = []
    cers = []
    wers = []
    iter_meter = IterMeter()
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, model = BRNN_two_hand_train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)
        val_loss, cer, wer = BRNN_two_hand_eval(model, device, val_loader, criterion, False)
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        cers.append(cer)
        wers.append(wer)
    end = time.time()
    BRNN_two_hand_eval(model, device, test_loader, criterion, True)
    save = 'y'
    if save == 'y':
        A = np.array(train_losses)
        B = np.array(test_losses)
        C = np.array(cers)
        D = np.array(wers)
        np.save(r'E:\My Models\both_hands_3D\BLSTM_2_hands_3D\BLSTMx2_clasifier3_dist_norm/BRNN_epoch_training_losses_.npy', A)
        np.save(r'E:\My Models\both_hands_3D\BLSTM_2_hands_3D\BLSTMx2_clasifier3_dist_norm/BRNN_epoch_test_losses.npy', B)
        np.save(r'E:\My Models\both_hands_3D\BLSTM_2_hands_3D\BLSTMx2_clasifier3_dist_norm/cers.npy', C)
        np.save(r'E:\My Models\both_hands_3D\BLSTM_2_hands_3D\BLSTMx2_clasifier3_dist_norm/wers.npy', D)
        print(f"Total time: {(end - start) / 60:.3f} minutes")
        torch.save(model.state_dict(), r'E:\My Models\both_hands_3D\BLSTM_2_hands_3D\BLSTMx2_clasifier3_dist_norm/model_weights.pth')
        fig = plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(1, epochs + 1), A)  # train loss (on epoch end)
        plt.plot(np.arange(1, epochs + 1), B)  # test loss (on epoch end)
        plt.title("model loss")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc="upper left")
        # 2nd figure
        plt.subplot(122)
        plt.plot(np.arange(1, epochs + 1), C)  # CER accuracy (on epoch end)
        plt.plot(np.arange(1, epochs + 1), D)  # WER accuracy (on epoch end)
        plt.title("CER & WER")
        plt.xlabel('epochs')
        plt.ylabel('cer wer')
        plt.legend(['cer', 'wer'], loc="upper left")
        title = r"E:\My Models\both_hands_3D\BLSTM_2_hands_3D\BLSTMx2_clasifier3_dist_norm/cer_wer_figure.png"
        title_eps = r"E:\My Models\both_hands_3D\BLSTM_2_hands_3D\BLSTMx2_clasifier3_dist_norm/cer_wer_figure.eps"
        plt.savefig(title, dpi=900)
        plt.savefig(title_eps, dpi=900)
        # plt.close(fig)
        plt.show()


def BRNN_two_hand_testing(test_dataset, learning_rate=5e-4, batch_size=20):

    hparams = {
        "n_rnn_layers": 2,
        "n_class": 28,
        "n_feats": 63,
        "dropout": 0.2,
        "learning_rate": learning_rate,
        "batch_size": batch_size
    }

    # experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Device {device}")
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing_2_hands(x),
                                  **kwargs)

    model = speech_recognition_model_brnn_two_hands(hparams['n_rnn_layers'], hparams['n_class'], hparams['n_feats'], hparams['dropout']).to(device)
    model.load_state_dict(torch.load('model_weights/model_weights_2_biLGRU_1fc_100EP_2_hands.pth'))
    # model.config.ctc_zero_infinity = True
    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    criterion = nn.CTCLoss(blank=27).to(device)
    # criterion = nn.CrossEntropyLoss()


    iter_meter = IterMeter()
    #for epoch in range(1, epochs + 1):
    BRNN_two_hand_eval(model, device, test_loader, criterion, True)

    #torch.save(model.state_dict(), 'model_weights.pth')


if __name__ == "__main__":
    signing_hand_solution = True
    all_bodypart_sol = False
    two_hands_solution = False
    CBRNN_sol = False
    train_cnn = False
    openpose_signing = False
    openpose_two_hand = False

    collect_hand_images = True
    if not os.path.isfile("my_lexicon.txt"):
        my_lexicon_file = open("my_lexicon.txt", "w+")
    else:
        my_lexicon_file = None

    #create_directories("abcdefghijklmnopqrstuvwxyz")

    if signing_hand_solution:
        learning_rate = 1e-4
        batch_size = 5
        epochs = 90
        training_set = data_extraction('signing_hand_data/train_set_data.npy', 'train', 0,
                                       'signing_hand_data/train_set.pkl', my_lexicon_file)
        #prepare_data_for_stt('train_set_data.npy','train', 0)
        validation_set = data_extraction('signing_hand_data/validation_set_data.npy', 'dev', 5455,
                                         'signing_hand_data/val_set.pkl', my_lexicon_file)
        #prepare_data_for_stt('validation_set_data.npy','dev',5455)
        test_set = data_extraction('signing_hand_data/test_set_data.npy', 'test', 6436,
                                   'signing_hand_data/test_set.pkl', my_lexicon_file)
        #prepare_data_for_stt('test_set_data.npy','test',6436)
        #all_sets = training_set + validation_set + test_set

        BRNN_signing_hand_solution(training_set, validation_set, test_set, learning_rate, batch_size, epochs)
        BRNN_signing_hand_testing(test_set, learning_rate, batch_size)

    if two_hands_solution:
        learning_rate = 1e-4
        batch_size = 5
        epochs = 50
        training_set = data_extraction_two_hands('two_hand_data/train_set_data_right_hands.npy',
                                                 'two_hand_data/train_set_data_left_hands.npy', 'train', 0,
                                                 'two_hand_data/train_set_dist.pkl', my_lexicon_file)
        # prepare_data_for_stt('train_set_data.npy','train', 0)
        validation_set = data_extraction_two_hands('two_hand_data/validation_set_data_right_hands.npy',
                                                   'two_hand_data/validation_set_data_left_hands.npy', 'dev', 5455,
                                                   'two_hand_data/val_set_dist.pkl', my_lexicon_file)
        # prepare_data_for_stt('validation_set_data.npy','dev',5455)
        test_set = data_extraction_two_hands('two_hand_data/test_set_data_right_hands.npy',
                                             'two_hand_data/test_set_data_left_hands.npy', 'test', 6436,
                                             'two_hand_data/test_set_dist.pkl', my_lexicon_file)
        # prepare_data_for_stt('test_set_data.npy','test',6436)
        # all_sets = training_set + validation_set + test_set

        print(len(training_set))
        print(len(validation_set))
        print(len(test_set))
        BRNN_two_hand_solution(training_set, validation_set, test_set, learning_rate, batch_size, epochs)
        BRNN_two_hand_testing(test_set, learning_rate, batch_size)

    if all_bodypart_sol:
        learning_rate = 1e-4
        batch_size = 5
        epochs = 50
        training_set = data_extraction_all_body_parts('two_hand_data/train_set_data_right_hands.npy',
                                                      'two_hand_data/train_set_data_left_hands.npy',
                                                      'two_hand_data/train_set_data_body.npy', 'train', 0,
                                                      'two_hand_data/train_set_all_body_parts.pkl', my_lexicon_file)
        # prepare_data_for_stt('train_set_data.npy','train', 0)
        validation_set = data_extraction_all_body_parts('two_hand_data/validation_set_data_right_hands.npy',
                                                        'two_hand_data/validation_set_data_left_hands.npy',
                                                        'two_hand_data/validation_set_data_body.npy', 'dev', 5455,
                                                        'two_hand_data/val_set_all_body_parts.pkl', my_lexicon_file)
        # prepare_data_for_stt('validation_set_data.npy','dev',5455)
        test_set = data_extraction_all_body_parts('two_hand_data/test_set_data_right_hands.npy',
                                                  'two_hand_data/test_set_data_left_hands.npy',
                                                  'two_hand_data/test_set_data_body.npy', 'test', 6436,
                                                  'two_hand_data/test_set_all_body_parts.pkl', my_lexicon_file)
        # prepare_data_for_stt('test_set_data.npy','test',6436)
        # all_sets = training_set + validation_set + test_set

        print(len(training_set))
        print(len(validation_set))
        print(len(test_set))


    # if CBRNN_sol:
    #     learning_rate = 1e-3
    #     batch_size = 5
    #     epochs = 100
    #
    #     hand_images_dir = r"C:\Users\Alekos\Desktop\Διπλωματική\MP&OP_hand_images"
    #     res_size = 224
    #     #mean = [0.485, 0.456, 0.406]
    #     #std = [0.229, 0.224, 0.225]
    #     train_transform = transforms.Compose([transforms.Resize([res_size, res_size]),
    #                                           transforms.ToTensor(),
    #                                           transforms.RandomRotation(5),
    #                                           transforms.RandomHorizontalFlip(0.5),
    #                                           transforms.RandomCrop(res_size, padding=10),
    #                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #                                           ])
    #
    #     val_transform = transforms.Compose([transforms.Resize([res_size, res_size]),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #                                         ])
    #
    #
    #     train_list,train_label,val_list,val_label,test_list,test_label = train_test_split()
    #     print(len(train_label))
    #     print(len(val_list))
    #     train_set = Dataset_CRNN(hand_images_dir, train_list, train_label, transform=train_transform)
    #     valid_set = Dataset_CRNN(hand_images_dir, val_list, val_label, transform=val_transform)
    #     test_set = Dataset_CRNN(hand_images_dir, test_list, test_label, transform=val_transform)
    #     #print(len(train_set))
    #
    #
    #     #CNN_BRNN_Solution(train_set, valid_set, learning_rate, batch_size, epochs)
    #     #CNN_BRNN_testing(test_set, learning_rate, batch_size)
    #
    #
    #    # if collect_hand_images:






class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        #self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        complete_video_path = os.path.join(path, selected_folder)
        for i in os.listdir(complete_video_path):
            complete_image_path = os.path.join(complete_video_path,i)
            image = Image.open(complete_image_path)

            if use_transform is not None:
                #plt.imshow(image)
                #plt.show()
                image = use_transform(image)



            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        #print("My shape",X.shape)
        #print(self.labels[index])
        #y = torch.Tensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor
        y = self.labels[index]
        # print(X.shape)
        return X, y


class Dataset_CRNN2(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, labels, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        # self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        #X = []
        complete_video_path = os.path.join(path, selected_folder)
        for idx,i in enumerate(os.listdir(complete_video_path)):
            complete_image_path = os.path.join(complete_video_path, i)
            image = Image.open(complete_image_path).convert('L')

            if use_transform is not None:
                # plt.imshow(image)
                # plt.show()
                image = use_transform(image)

            #X.append(image)
            if idx == 0:
                X = image
            else:
                X = torch.cat((X,image),dim = 2)
        #X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)  # (input) spatial images
        X = X.transpose(1,2)
        X = torch.squeeze(X)
        # transform = transforms.ToPILImage()
        # img = transform(X)
        # img.show()
        # print("My shape", X.shape)
        # print(self.labels[index])
        # y = torch.Tensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor
        y = self.labels[index]
        # print(X.shape)
        return X, y



class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        #print("Size before transpose", x.shape)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        #print("Size after transpose", x.shape)
        x = self.layer_norm(x)
        #print("Size after fully connected", x.shape)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = activation_function.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = activation_function.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class EncoderCNN(nn.Module):
    def __init__(self, img_x=90, img_y=120, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (7, 7), (5, 5), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        with torch.no_grad():
            self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
            self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
            self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
            self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
             nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
             nn.BatchNorm2d(self.ch2, momentum=0.01),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
             nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
             nn.BatchNorm2d(self.ch3, momentum=0.01),
             nn.ReLU(inplace=True),
             #nn.MaxPool2d(kernel_size=2),
        )

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
        #     nn.BatchNorm2d(self.ch4, momentum=0.01),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2),
        # )

        self.drop = nn.Dropout2d(self.drop_p)
        #self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, self.CNN_embed_dim)   # fully connected layer, output k classes
        #self.fc2 = nn.Linear(self.fc_hidden1, self.CNN_embed_dim)
        #self.fc3 = nn.Linear(self.fc_hidden2, )   # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        #print(x_3d.shape)
        for t in range(x_3d.size(1)):
            x = self.conv1(x_3d[:, t, :, :, :])
            #print(f"After conv1 {x.shape}")
            x = self.conv2(x)
            #print(f"After conv2 {x.shape}")
            x = self.conv3(x)
            #print(f"After conv3 {x.shape}")
            #x = self.conv4(x)
            #print(f"After conv4 {x.shape}")
            x = x.view(x.size(0), -1)           # flatten the output of conv
           # print(f"After flatten {x.shape}")
            x = activation_function.relu(self.fc1(x))
            #x = self.drop(x)
            #x = F.relu(self.fc2(x))
            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq


def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

