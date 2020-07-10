import os
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class HeadPoseModel:
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
        self.exec_net = self.core.load_network(network=self.network, device_name=self.device,num_requests=self.num_requests)
        
        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

        return self.exec_net

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_frame = self.preprocess_input(image)
        self.exec_net.start_async(request_id=self.request_id,inputs={self.input: processed_frame})
        self.exec_net.requests[0].get_perf_counts()

        if self.mode == 'async':
            self.exec_net.requests[0].wait()
            return self.preprocess_output(self.exec_net.requests[0].outputs)
        else:
            if self.exec_net.requests[0].wait(-1) == 0:
                return self.preprocess_output(self.exec_net.requests[0].outputs)

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

        image_resized = cv2.resize(image, (model_input_shape[3], model_input_shape[2]))
        p_frame = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return p_frame

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        return np.array([outputs['angle_y_fc'][0][0], outputs['angle_p_fc'][0][0], outputs['angle_r_fc'][0][0]])

