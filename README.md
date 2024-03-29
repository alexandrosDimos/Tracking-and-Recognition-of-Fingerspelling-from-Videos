# Tracking-and-Recognition-of-Fingerspelling-from-Videos
This repository contains the code developed for my Thesis project. In this Thesis we examined how machine learning techniques can be utilized in sign language recognition. We mainly focused on American Sign Language.
## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Thsesis Paper](#thesis-paper)


## Project Description

We will be using the frame sequences from online videos, provided by the ChicagoFSWild dataset. The videos depict people facing a camera and performing fingerspelling. We focus mainly on the hand and finger motion and try to distinguish the signing hand. To track the hands we use MediaPipe and OpenPose, which are able to produce 3D and 2D skeleton coordinates of specified body parts such as hands, body, head etc. We proceed by comparing the performance of these libraries and then use the provided coordinates to determine the signing hand. Both 2D and 3D skeleton coordinates are used as input to our machine learning models in various combinations. The models use the aforementioned coordinates within a bidirectional recurrent neural network.

## Features

The project can be devided in three smaller parts.
- Extraction of 3D and 2D body part skeletons with assisatnce from Mediapipe and OpenPose.
- A comparison between MediaPipe and OpenPose, based on each one's detection accuracy.
- Developing the architecture of the neaural networks, training them and make our conclusions.

## Extraction of 3D and 2D body part skeletons with assisatnce from Mediapipe and OpenPose.
During this phase we focus on how to extract body features from the video frames in our Dataset. This projet started with the intention of only using MediaPipe as our computer vision library. Although certain limitations and further curiosity drove us to include OpenPose. For this reason we developed to separate sets of tools:
- MyMediaPipeToolkit
- MyOpenPoseToolkit

Each component contains classes created specifically for each library, some of the most interesting algorithms we developed were:
- Distinguish the hand that signs automatically.
- Discard culprit body part detections, a phenomemnon present especially with OpenPose.
- Being able to distinguish the person performing ASL among multiple detections.
Using these modules we also managed to createa an algorithm, where OpenPose and MediaPipe worked in accordance with each other in order to boost the number and the quality of detections. Unfortunately MediaPipe could not at the time detect more then one person in a frame, a liability OpenPose managed to cover.

## A comparison between MediaPipe and OpenPose, based on each one's detection accuracy.
The creators of the dataset included also BBox. A subset including manually drawn bounding boxes around the hands of the signers and idicators on the hand siging. Based on this, we evaluated the performance of the computer vision libraries and our own algorithms, using the metrics below:
- Intersection over Union
- Precision
- Recall
- F1-Score


## Developing the architecture of the neaural networks, training them and make our conclusions.
The final part of our project can be separated in:
- Normalization techniques.
- Developing a neural network architecture. The main NN componenets and architectures we used are:
  - LSTM
  - GRU
  - Bidirectional RNN
- Decoding the output of the NN architetcure. For that we utilized:
  - Greedy Decoder
  - Beam search decoder. The beam search algorithm we used and experimented on can be found here https://github.com/githubharald/CTCDecoder.
- Evaluating the results. For that purpose we used levenshtein distance to measure the difference between the real and the predicted words. The algorithm we used can also be found here https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/.

## Disclaimer
All of the work here was done locally on my machine, so every path written should be changed before it can be reused. I know my code and workflow can be futher optimized for better performance, but the purpose of this project was to develop neural networks for a specific purpose, so more attention was given to the final results.
    
## Thesis paper
Anyone looking for a more detailed and deeper understanding of the project as well as the final results and conclusions of my work can take a look:
https://www.e-ce.uth.gr/wp-content/uploads/formidable/59/Dimos_alexandros.pdf

