# Human Activity Recognition
### Description :
This project is made with a perspective of Recognising Human Activities. The human activity recognition model was trained on Kinetics 400 Dataset.
To learn more about this dataset and model used refer Kay et al.’s 2017 paper, [The Kinetics Human Action Video Dataset.](https://arxiv.org/abs/1705.06950) and Hara et al.’s 2018 CVPR paper, [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/abs/1711.09577)
The authors [Kensho Hara, Hirokatsu Kataoka, Yutaka Satoh](https://arxiv.org/abs/1711.09577) have mentioned in their 2018 CVPR paper, how existing state-of-the-art 2D architectures (such as ResNet, ResNeXt, DenseNet, etc.) can be extended to video classification via 3D kernels.

### Prerequisites :
  ###### Required 
  - Python Programming Language 
  - Convolutional Neural Network 

  ###### Not compulsory(given below), but pior knowledge would be beneficial 
  - [ONXX-Based Deep Learning models](https://github.com/onnx/models) 
  - [OpenCV for Python](https://opencv.org/)

### Installations :
 - [Install Python](https://www.python.org/downloads/)<br/>
 - [Install OpenCV for windows](https://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html) <br/>
 - [Install OpenCV for Ubuntu](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html) <br/>
- Execute the following command in terminal: <br/>
` pip install numpy` <br/>

### Directory Structure :
![](https://github.com/techycs18/human-activity-recognition/blob/master/Direcory%20Stucture.png)

### External download :
- You need to dowwload a **`resnet-34_kinetics.onnx`** model from [**here**](https://www.dropbox.com/s/065l4vr8bptzohb/resnet-34_kinetics.onnx?dl=1) and once downloaded drop it inside the `model` directory of our project (shown in directory structure)

### Setup :
1. I have added a video example for testing in `test` directory
2. If you want to test your own video file be sure to add it in `test` folder 
3. Now, inside `recognise_human_activity.py` constructor set instance variable `VIDEO_PATH` to you file path. 
4. Otherwise, if you want test the model on using web-camera live video just set `self.VIDEO_PATH = None`  
- Once your setup is done run the following to execute code:
```
python recognise_human_activity.py
```
### Screenshots :
##### Playing Keyboard -
![-](https://github.com/techycs18/human-activity-recognition/blob/master/screenshots/playing_keyboard.png)
##### High Kick -
![-](https://github.com/techycs18/human-activity-recognition/blob/master/screenshots/high_kick.png)
##### Push Ups -
![-](https://github.com/techycs18/human-activity-recognition/blob/master/screenshots/pushups.png)
#### Author 
- [Chaitanya Sonavane](https://www.linkedin.com/in/chaitanya-sonavane-3766521a0/) [July 2020] 

#### Acknowledments :
- Kay et al.’s 2017 paper, [The Kinetics Human Action Video Dataset.](https://arxiv.org/abs/1705.06950)
- Hara et al.’s 2018 CVPR paper, [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/abs/1711.09577)
- OpenCV's [Action Recognition Example](https://github.com/opencv/opencv/blob/master/samples/dnn/action_recognition.py) 
