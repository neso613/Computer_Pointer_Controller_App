# Computer Pointer Controller

Computer Pointer Controller application used to control mouse pointer movement by eye gaze and head pose of human being. The project's main aim is to deploy multiple model altogether using OpenVino ToolKit. I have used 4 pre-trained model provided by Open Model Zoo. This app is an integration of face detection model, head-pose estimation model, facial landmarks model and gaze estimation deep learning model.

## Project Set Up and Installation
- Download *[OpenVino ToolKit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)* and *[install](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)* it.

- *[Clone the repository]()*

- Initialize the openVINO environment using this command
  >source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

#### Download Models used for this application using following commands

*Face Detection Model*
>python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"

*Facial Landmark Detection Model*
>python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"

*HeadPose Estimation Model*
>python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"

*Gaze Estimation Model*
>python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"

These models can also be downloaded via *[Open Model Zoo](https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/)*.

## Demo
Open a new terminal and run the following commands -
- Go to project directory and go to <strong>src</strong> folder
    > cd <project-repo-path>/src

- Run main.py for CPU
    >python main.py -f <Path of xml file of face detection model> \
-lm <Path of xml file of facial landmarks detection model> \
-hp <Path of xml file of head pose estimation model> \
-g <Path of xml file of gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam> 

- Run main.py for GPU
    >python main.py -f <Path of xml file of face detection model> \
-lm <Path of xml file of facial landmarks detection model> \
-hp <Path of xml file of head pose estimation model> \
-g <Path of xml file of gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam> \
-d GPU

- Run main.py for FPGA
    >python main.py -f <Path of xml file of face detection model> 
-lm <Path of xml file of facial landmarks detection model> 
-hp <Path of xml file of head pose estimation model> 
-g <Path of xml file of gaze estimation model> 
-i <Path of input video file or enter cam for taking input video from webcam> 
-d HETERO:FPGA,CPU

- Run main.py for MYRIAD
    >python main.py -f <Path of xml file of face detection model> \
-lm <Path of xml file of facial landmarks detection model> \
-hp <Path of xml file of head pose estimation model> \
-g <Path of xml file of gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam> \
-d HETERO:MYRIAD,CPU

## Documentation
#### Details of Models
- *[Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)*
- *[Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)*
- *[Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)*
- *[Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)*

#### Command Line Arguments used in this app
- -h : Get the information about all the command line arguments
- -f (required) : Specify the path of Face Detection model's xml file
- -g (required) : Specify the path of Gaze Estimation model's xml file
- -lm (required) : Specify the path of Facial Landmark Detection model's xml file
- -hp (required) : Specify the path of Head Pose Estimation model's xml file
- -i (required) : Specify the path of input video file or enter cam for taking input video from webcam
- -l (optional) : Specify the absolute path of cpu extension if some layers of models are not supported on the device.
- -prob (optional) : Specify the probability threshold for face detection model to detect the face accurately from video frame.
- -flags (optional) : Specify the flags from fd, fld, hp, ge if you want to visualize the output of corresponding models of each frame (write flags with space seperation. Ex:- -flags fd fld hp).
- -d (optional) : Specify the target device to infer the video file on the model. Suppoerted devices are: CPU, GPU, FPGA (For running on FPGA - used HETERO:FPGA,CPU), MYRIAD.

#### Python Files Details
- **face_detection.py** contains code to preprocess input, perform inference and process output on video stream input.
- **facial_landmarks_detection.py** have code to find facial landmark points on face. It take the deteted face as input and detect the eye landmarks, postprocess the outputs.
- **head_pose_estimation.py** takes the detected face as input, preprocessed it and perform inference on it and detect the head postion by predicting yaw - roll - pitch angles, postprocess the outputs.
- **gaze_estimation.py** takes the left eye, rigt eye, head pose angles as inputs, preprocessed it, perform inference and predict the gaze vector, postprocess the outputs.
- **input_feeder.py** have InputFeeder class which initialize VideoCapture as per the user argument and return the frames one by one.
- **mouse_controller.py** have MouseController class which take x, y coordinates value, speed, precisions and according these values it moves the mouse pointer by using pyautogui library.
- **main.py** is main file for running the app.

#### Directory Structure of the Project
![picture](directoryStructure.png)

## Benchmarks
#### CPU
| Precision      | Inference Time | Model Loading Time | FPS |
| -------------- | -------------- | ------------------ | --- |
| FP32           | 65             | 1.8                | 9   |
| FP16           | 79             | 1.9                | 8   |
| INT8           | 72             | 1.4                | 6   |

#### GPU
| Precision      | Inference Time | Model Loading Time | FPS |
| -------------- | -------------- | ------------------ | --- |
| FP16           | 76             | 59.2               | 9   |

#### FPGA
| Precision      | Inference Time | Model Loading Time | FPS |
| -------------- | -------------- | ------------------ | --- |
| FP32           | 103            | 25                 | 4   |
| FP16           | 129            | 16                 | 6   |
| INT8           | 162            | 8                  | 3   |

## Results
- Intel Atom x7-E3950 UP2 GPU takes so much time to load the model.
- FPGA works very slow as it is taking high inference time.
- Intel Core i5-6500TE CPU provides simillar performance like 65000TE GPU.
- Priority is to run the application on Intel Core i5-6500TE GPU.

## Stand Out Suggestions
**IP Web Cam:** - I assume that not all computers are connected via webcam so to get the realtime feel of the project I have used feed of IP WEB CAM to get the live feed from an android app.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
- If there is more than one face detected, it extracts only one face and do inference on it and ignoring other faces.
- Also, If multiple faces detected. It only proceed with one face. Multiple face are not supported.
- Moving Mouse Pointer out of the maximum window width.
