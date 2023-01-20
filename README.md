# PET DETECTION & IDENTIFICATION SYSTEM USING YOLOv5

## INTRODUCTION

YoloV5 is an objection detection which is already trained on a large dataset and thousands of classes. Yolov5 is basically used for object detection which is used for detecting objects in images as well as videos. We can directly train the YoloV5 model for object detection to detect the objects. In this project YOLOv5 is being used to detect and identify cats and dogs in real time.

## STEPS TO REPLICATE THIS PROJECT:

### 1. Clone the github repository by typing the following commands
```
# Clone
git clone https://github.com/CHRISTInfotech/AI-IoT-Basic-Projects.git
cd CAT & DOG IDENTIFICATION SYSTEM USING YOLOv5

# Install yolov5
git clone https://github.com/ultralytics/yolov5  
cd yolov5
pip install -r requirements.txt

```
### 2. Open the link to the dataset in the readme section in the folder. 

  - This will leads to a kaggle file having the dataset of about 3000 cats and dogs image.
  - Download the file and paste it inside the folder ```\DATASET```.
  - For creating your own dataset follow the steps below:
    - Collect pictures of both cat and dogs.
    - Next step is to annotate them by bounding boxes.
    - Use the [make sense](https://www.makesense.ai/) or [roboflow](https://docs.roboflow.com/annotate) to annotate.
    - Split the dataset into train and Val. The path to be followed is mentioned in the step 3.
          
### 3. Create a YAML file which contains the path to the dataset, Training & Validation and specific the number of classes and class names

  - This has been done in a file called pet_directory.yaml.
  - open and change the path according to your dataset destination.
  
      ```bash
    ├── dataset
    │   └── pet
    │       ├── train
    │       │     ├── images
    │       │     └── labels
    │       │ 
    │       └── val
    │             ├── images
    └──           └── labels
    ```
 
  
### 4. Train the model using the command given below

  - Open ```train.ipynb``` script for training the model from scratch. 
    ```
    python train.py --img 640 --batch 16 --epochs 10 --data ../pet_directory.yaml --weights yolov5s.pt --workers 0
    ```
### 5. Prediction
   - After training done the results would be obtained in the runs folder
   - open he folder and locate the weights for the current experiment
   - Copy the folder path and paste in the ```--weight``` section given below
   - While runnning this code make the port of the uno connected to the usb so that the error does't occurs
   - General options for detect.py
      
      ```
      python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
                                                     
                                                     
      python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
      ```

   - If you train your own model, use the following command for detection:
      ```
      python detect.py --source ../input.mp4 --weights runs/train/exp/weights/best.pt --conf 0.2
      ```
   
   - Or you can use the pretrained model located in ```models``` folder for detection as follows:
      ```
      python detect.py --source ../input.mp4 --weights ../models/best.pt --conf 0.2
      ```

## Prediction Results
The pet detection results were fairly good even though the model was trained only for a few epochs. The [authors](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results) who created YOLOv5 recommend using about 0-10% background images to help reduce false positives. 

### **Training Dataset**

   <p align="center">
  <img src="https://user-images.githubusercontent.com/114398468/213620313-9ee79f0e-124d-4832-8fb1-104bf839a172.jpg" />
</p>

### **Prediction**

  <p align="center">
  <img src="https://user-images.githubusercontent.com/114398468/213620140-25e97f54-2349-45d3-aaae-3397c27a1bb3.jpg" />
</p>

  
### **Evaluation Metrics**

  <p align="center">
  <img src="https://user-images.githubusercontent.com/114398468/213621041-f1823949-d90f-4b02-9ea3-81bda316316e.png" />
</p>


## OUTPUT:
- The output obtained from this overall project is given in the following link:-
  
  https://user-images.githubusercontent.com/114398468/213620787-eb9d7fea-7c80-4490-9c5c-f73d19dfa83b.mp4

- IMAGE:

  <p align="center">
  <img src="https://user-images.githubusercontent.com/114398468/213619458-f2108854-24ae-4eb6-a0f8-ea72796082ab.png" />
</p>


## REFERENCE:
- [https://www.hackster.io/innovation4x/early-fire-detection-using-ai-dd27bf](https://www.kaggle.com/c/dogs-vs-cats)
- https://ultralytics.com/yolov5
- https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results
- https://github.com/ultralytics/yolov5.git







