# glucoma-detection-using-cnn
## Overview
A simple Python Flask based Web Application for detecting glucoma in early stages.
The app uses inception model for classification using Tensorflow.
You will find pretrained inception model in Tensorflow directory.

## Run

```
> Install Requirements
1) pip install -r requirements.txt

> Retrain Inception - Final Layer
2) python retrain.py --bottleneck_dir=Tensorflow/Bottlenecks --how_many_training_steps 2000 --model_dir=Tensorflow/inception --output_graph=Tensorflow/Graph.pb --output_labels=Tensorflow/Labels.txt --image_dir=Tensorflow/dataset

> Test Inception
3) python label.py testImg.jpg #testImg can be any any image you want to test

> Run Flask Sever
4) python server.py

```
