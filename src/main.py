import cv2
import os
import logging
import numpy as np
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FaceLandmarkModel
from gaze_estimation import EyeGazeModel
from head_pose_estimation import HeadPoseModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help="Specify Path to .xml file of Face Detection model.")
    parser.add_argument("-lm", "--facelandmarkmodel", required=True, type=str,
                        help="Specify Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str,
                        help="Specify Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    
    return parser


def main():
    args = build_argparser().parse_args()
    logger = logging.getLogger()

    if args.input_type == 'video' or args.input_type == 'image':
        extension = str(args.input).split('.')[1]
        feeder = InputFeeder(args.input_type, args.input)
    elif args.input_type == 'cam':
        feeder = InputFeeder(args.input_type)

    mc = MouseController("medium", "fast")
    feeder.load_data()

    face_model = FaceDetectionModel(args.facedetectionmodel, args.device, args.cpu_extension)
    face_model.check_model()

    landmark_model = Landmark_Model(args.facelandmarkmodel, args.device, args.cpu_extension)
    landmark_model.check_model()

    gaze_model = Gaze_Estimation_Model(args.gazeestimationmodel, args.device, args.cpu_extension)
    gaze_model.check_model()

    head_model = Head_Pose_Model(args.headposemodel, args.device, args.cpu_extension)
    head_model.check_model()

    face_model.load_model()
    logger.info("Face Detection Model Loaded...")

    landmark_model.load_model()
    logger.info("Landmark Detection Model Loaded...")
    
    head_model.load_model()
    logger.info("Head Pose Detection Model Loaded...")

    gaze_model.load_model()
    logger.info("Gaze Estimation Model Loaded...")

    logger.info('All Models are loaded\n\n')
    out = cv2.VideoWriter('output_video.mp4', 0x00000021, 30, (500,500))

 
    frame_count = 0
    for ret, frame in feeder.next_batch():
        if not ret:
            break
            frame_count += 1
            
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))
        key = cv2.waitKey(60)
        faceROI = None
           
        if True:
            faceROI, box = FaceDetectionModel.predict(frame.copy(),args.prob_threshold)
            if faceROI is None:
                logger.error("Unable to detect the face.")
                if key==27:
                    break
                continue

            (lefteye_x, lefteye_y), (righteye_x, righteye_y), eye_coords, left_eye, right_eye = FaceLandmarkModel.predict(faceROI.copy(), EYE_ROI=10)
            head_position = HeadPoseModel.predict(faceROI.copy())
            new_mouse_coord, gaze_vector = EyeGazeModel.predict(left_eye.copy(), right_eye.copy(), head_position)

            if (not len(previewFlags)==0):
                preview_frame = frame.copy()
                if 'fd' in previewFlags:
                    #cv2.rectangle(preview_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 3)
                    preview_frame = croppedFace
                if 'fld' in previewFlags:
                    cv2.rectangle(croppedFace, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 3)
                    cv2.rectangle(croppedFace, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 3)
                    #preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = croppedFace
                if 'hp' in previewFlags:
                    cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0],hp_out[1],hp_out[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
                if 'ge' in previewFlags:
                    x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                    le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                    cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                    re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                    cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                    croppedFace[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
                    croppedFace[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
                    #preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = croppedFace
                #cv2.imshow("visualization",cv2.resize(preview_frame,(500,500)))
                out.write(frame)
        
            if frame_count%5==0:
                mc.move(new_mouse_coord[0],new_mouse_coord[1])    
            if key==27:
                break
            
    logger.error("VideoStream ended...")
    out.release()
    cv2.destroyAllWindows()
    inputFeeder.close()
     
    

if __name__ == '__main__':
    args=parser.parse_args()
    main(args) 

 