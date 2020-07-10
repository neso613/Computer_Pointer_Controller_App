import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class FaceLandmarkModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split(".")[0]+'.bin'
        self.core = None
        self.network = None
        self.exec_net = None
        
        self.input = None
        self.output = None
        self.mode = 'async'
        self.request_id = 0

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()

        self.network = self.core.read_network(model=self.model_structure, weights=self.model_weights)
        self.exec_net = self.core.load_network(self.network, self.device)
        
        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

        return self.exec_net

    def predict(self, image, EYE_ROI):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        
        processed_frame = self.preprocess_input(image)
        self.exec_net.start_async(request_id=self.request_id,inputs={self.input: processed_frame})
        self.exec_net.requests[0].get_perf_counts()

        if self.mode == 'async':
            self.exec_net.requests[0].wait()
            result = self.exec_net.requests[0].outputs[self.output]
            return self.preprocess_output(result, image, EYE_ROI)

        else:
            if self.exec_net.requests[0].wait(-1) == 0:
                result = self.exec_net.requests[0].outputs[self.output]
                return self.preprocess_output(result, image, EYE_ROI)

    def check_model(self):
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        # if len(unsupported_layers) > 0:
        #     print("Please check extention for these unsupported layers =>" + str(unsupported_layers))
        #     exit(1)
        # print("All layers are supported !!")

        if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                print("Adding cpu_extension")
                self.core.add_extension(self.extensions, self.device)
                supported_layers = self.core.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("After adding the extension still unsupported layers found")
                    exit(1)
                print("After adding the extension the issue is resolved")
            else:
                print("Give the path of cpu extension")
                exit(1)

    def preprocess_input(self, image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        
        model_input_shape = self.network.inputs[self.input].shape

        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_cvt, (model_input_shape[3], model_input_shape[2]))
        p_frame = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return p_frame

    def preprocess_output(self, outputs, image, EYE_ROI):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        left_x = outputs[0][0].tolist()[0][0]
        left_y = outputs[0][1].tolist()[0][0]
        right_x = outputs[0][2].tolist()[0][0]
        right_y = outputs[0][3].tolist()[0][0]

        box = (left_x, left_y, right_x, right_y)

        h, w = image.shape[0:2]
        box = box * np.array([w, h, w, h])
        box = box.astype(np.int32)

        (lefteye_x, lefteye_y, righteye_x, righteye_y) = box

        leftxmin = lefteye_x - EYE_ROI
        leftymin = lefteye_y - EYE_ROI
        leftxmax = lefteye_x + EYE_ROI
        leftymax = lefteye_y + EYE_ROI

        rightxmin = righteye_x - EYE_ROI
        rightymin = righteye_y - EYE_ROI
        rightxmax = righteye_x + EYE_ROI
        rightymax = righteye_y + EYE_ROI

        left_eye = image[leftymin:leftymax, leftxmin:leftxmax]
        right_eye = image[rightymin:rightymax, rightxmin:rightxmax]
        eye_coords = [[leftxmin, leftymin, leftxmax, leftymax], [rightxmin, rightymin, rightxmax, rightymax]]

        # cv2.rectangle(image,(lefteye_x,lefteye_y),(righteye_x,righteye_y),(255,0,0))
        return (lefteye_x, lefteye_y), (righteye_x, righteye_y), eye_coords, left_eye, right_eye

