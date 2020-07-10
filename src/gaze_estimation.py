import cv2
import numpy as np
from openvino.inference_engine import IECore
import math
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class EyeGazeModel:
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
        self.model_weights = self.model_name.split('.')[0]+'.bin'

        self.core = None
        self.network = None
        self.exec_net = None

        self.mode = 'async'
        self.request_id = 0

        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()

        self.network = self.core.read_network(model=self.model_structure, weights=self.model_weights)
        self.exec_net = self.core.load_network(network=self.network, device_name=self.device,num_requests=self.num_requests)

        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

        return self.exec_net

    def predict(self, left_eye, right_eye, head_position):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_left_eye = self.preprocess_input(left_eye)
        processed_right_eye = self.preprocess_input(right_eye)
        self.exec_net.start_async(request_id=self.request_id,
                                      inputs={'left_eye_image': processed_left_eye,
                                              'right_eye_image': processed_right_eye,
                                              'head_pose_angles': head_position})
        self.exec_net.requests[0].get_perf_counts()

        if self.exec_net.requests[0].wait(-1) == 0:
            result = self.exec_net.requests[0].outputs[self.output]
            new_mouse_coord, gaze_vector = self.preprocess_output(result, head_position)
            return new_mouse_coord, gaze_vector

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

        model_input_shape = self.network.inputs[self.input_name[1]].shape
        image_resized = cv2.resize(image, (model_input_shape[3], model_input_shape[2]))
        p_frame = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return p_frame

    def preprocess_output(self, output, head_position):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        gaze_vector = output[self.output_names[0]].tolist()[0]
        rollValue = hpa[2] 
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        newx = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        newy = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue
        return (newx,newy), gaze_vector

    
