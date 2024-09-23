# ObjectDetectionFromScratch
This is my own implementation of object detection from scratch

My goal here is to write objection detection neural network to 
learn their architectures and training basics for myself, however 
I had simplicity in mind for myself or if someone else want to use it.
If you know some basic pytorch this code will be extremely easy for you.

### Download Data
First download data from CoCo website https://cocodataset.org/#download
and unzip it to data folder under repository so basically you should have following structure:
```
-data
    -annotations
    -train2017
    -val2017
-src
    ... #
```
Where train2017 and val2017 are folders with images



### Run train
```commandline
python3 src/train_kenny_net_cnn
python3 src/train_kenny_net_detr
```

Here, KennyNetCNN with EfficientNet backbone and 
feature pyramid network with outputs similar to yolo (with anchors and such)

KennyNetDetR is EfficientNet backbone and self-made swin transformer from scratch


### Run eval
```commandline
python3 src/eval_kenny_net_cnn
python3 src/eval_kenny_net_detr
```