import models as nnmodels
import utilities as nnutils
import torch
import torch.optim as optim
import os
import joblib
import torch.onnx
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import sys
import numpy as np
from onnx import numpy_helper



def pt_2_nnet(destination_path_onnx, destination_path_nnet, model,device, batch_size, seq_len):
    print("WARNING: Using the default values of input bounds and normalization constants")
    export_onnx(destination_path_onnx, model,device, batch_size, seq_len)
    onnx2nnet(destination_path_onnx,nnetFile=destination_path_nnet)

def writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,fileName):
    '''
    Write network data to the .nnet file format

    Args:
        weights (list): Weight matrices in the network order 
        biases (list): Bias vectors in the network order
        inputMins (list): Minimum values for each input
        inputMaxes (list): Maximum values for each input
        means (list): Mean values for each input and a mean value for all outputs. Used to normalize inputs/outputs
        ranges (list): Range values for each input and a range value for all outputs. Used to normalize inputs/outputs
        fileName (str): File where the network will be written
    '''
    
    #Open the file we wish to write
    with open(fileName,'w') as f2:

        #####################
        # First, we write the header lines:
        # The first line written is just a line of text
        # The second line gives the four values:
        #     Number of fully connected layers in the network
        #     Number of inputs to the network
        #     Number of outputs from the network
        #     Maximum size of any hidden layer
        # The third line gives the sizes of each layer, including the input and output layers
        # The fourth line gives an outdated flag, so this can be ignored
        # The fifth line specifies the minimum values each input can take
        # The sixth line specifies the maximum values each input can take
        #     Inputs passed to the network are truncated to be between this range
        # The seventh line gives the mean value of each input and of all outputs
        # The eighth line gives the range of each input and of all outputs
        #     These two lines are used to map raw inputs to the 0 mean, unit range of the inputs and outputs
        #     used during training
        # The ninth line begins the network weights and biases
        ####################
        f2.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")

        #Extract the necessary information and write the header information
        numLayers = len(weights)
        inputSize = weights[0].shape[0]
        outputSize = len(biases[-1])
        maxLayerSize = inputSize
        
        # Find maximum size of any hidden layer
        for b in biases:
            if len(b)>maxLayerSize:
                maxLayerSize = len(b)
        # Write data to header 
        f2.write("%d,%d,%d,%d,\n" % (numLayers,inputSize,outputSize,maxLayerSize) )
        f2.write("%d," % inputSize )
        for b in biases:
            f2.write("%d," % len(b) )
        f2.write("\n")
        f2.write("0,\n") #Unused Flag

        # Write Min, Max, Mean, and Range of each of the inputs and outputs for normalization
        f2.write(','.join(str(inputMins[i])  for i in range(inputSize)) + ',\n') #Minimum Input Values
        f2.write(','.join(str(inputMaxes[i]) for i in range(inputSize)) + ',\n') #Maximum Input Values                
        f2.write(','.join(str(means[i])      for i in range(inputSize+1)) + ',\n') #Means for normalizations
        f2.write(','.join(str(ranges[i])     for i in range(inputSize+1)) + ',\n') #Ranges for noramlizations

        ##################
        # Write weights and biases of neural network
        # First, the weights from the input layer to the first hidden layer are written
        # Then, the biases of the first hidden layer are written
        # The pattern is repeated by next writing the weights from the first hidden layer to the second hidden layer,
        # followed by the biases of the second hidden layer.
        ##################
        for w,b in zip(weights,biases):
            for j in range(w.shape[1]):
                for i in range(w.shape[0]):
                    f2.write("%.5e," % w[i][j]) #Five digits written. More can be used, but that requires more more space.
                f2.write("\n")
                
            for i in range(len(b)):
                f2.write("%.5e,\n" % b[i]) #Five digits written. More can be used, but that requires more more space.

def export_onnx(path, model,device, batch_size, seq_len):#batch_size = 1 - seq_len = 35
    model.eval()
    #dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    dummy_input = torch.Tensor(64 ,1, 28, 28)
    torch.onnx.export(model, dummy_input, path, input_names=['input'], output_names=['output'])
    print('The model is also exported in ONNX format at {}'.
        format(os.path.realpath(path)))

def onnx2nnet(onnxFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName=""):
    '''
    Write a .nnet file from an onnx file
    Args:
        onnxFile: (string) Path to onnx file
        inputMins: (list) optional, Minimum values for each neural network input.
        inputMaxes: (list) optional, Maximum values for each neural network output.
        means: (list) optional, Mean value for each input and value for mean of all outputs, used for normalization
        ranges: (list) optional, Range value for each input and value for range of all outputs, used for normalization
        inputName: (string) optional, Name of operation corresponding to input.
        outputName: (string) optional, Name of operation corresponding to output.
    '''
    
    if nnetFile=="":
        nnetFile = onnxFile[:-4] + 'nnet'

    model = onnx.load(onnxFile)
    graph = model.graph

    if not inputName:
        #assert len(graph.input)==1 #prova a generare un dummy_input di 1
        inputName = graph.input[0].name
    if not outputName:
        #assert len(graph.output)==1
        outputName = graph.output[0].name
    
    # Search through nodes until we find the inputName.
    # Accumulate the weight matrices and bias vectors into lists.
    # Continue through the network until we reach outputName.
    # This assumes that the network is "frozen", and the model uses initializers to set weight and bias array values.
    weights = []
    biases = []

    # Loop through nodes in graph
    for node in graph.node:
        
        # Ignore nodes that do not use inputName as an input to the node
        if inputName in node.input:
            
            # This supports three types of nodes: MatMul, Add, and Relu
            # The .nnet file format specifies only feedforward fully-connected Relu networks, so
            # these operations are sufficient to specify nnet networks. If the onnx model uses other 
            # operations, this will break.
            if node.op_type=="MatMul":
                assert len(node.input)==2

                # Find the name of the weight matrix, which should be the other input to the node
                weightIndex=0
                if node.input[0]==inputName:
                    weightIndex=1
                weightName = node.input[weightIndex]

                
                # Extract the value of the weight matrix from the initializers
                weights+= [numpy_helper.to_array(inits).T for inits in graph.initializer if inits.name==weightName]
                
                # Update inputName to be the output of this node
                inputName = node.output[0]
                
            elif node.op_type=="Add":
                assert len(node.input)==2
                # Find the name of the bias vector, which should be the other input to the node
                biasIndex=0
                if node.input[0]==inputName:
                    biasIndex=1
                biasName = node.input[biasIndex]
                
                # Extract the value of the bias vector from the initializers
                biases+= [numpy_helper.to_array(inits) for inits in graph.initializer if inits.name==biasName]
                
                # Update inputName to be the output of this node
                inputName = node.output[0]
                
            # For the .nnet file format, the Relu's are implicit, so we just need to update the input
            elif node.op_type=="Relu":
                inputName = node.output[0]
            # If there is a different node in the model that is not supported, through an error and break out of the loop
            else:
                print("Node operation type %s not supported!"%node.op_type)
                # weights = []
                # biases=[]
                # break
                
            # Terminate once we find the outputName in the graph
            #if outputName == inputName:
                #break
    # Check if the weights and biases were extracted correctly from the graph
    if outputName==inputName and len(weights)>=0 and len(weights)==len(biases):
        
        inputSize = weights[0].shape[0]
        
        # Default values for input bounds and normalization constants
        if inputMins is None: inputMins = inputSize*[np.finfo(np.float32).min]
        if inputMaxes is None: inputMaxes = inputSize*[np.finfo(np.float32).max]
        if means is None: means = (inputSize+1)*[0.0]
        if ranges is None: ranges = (inputSize+1)*[1.0]
            
        # Print statements
        print("Converted ONNX model at %s"%onnxFile)
        print("    to an NNet model at %s"%nnetFile)
        
        # Write NNet file
        writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,nnetFile)
        print('The model is also converted in NNET format at {}'.
                format(os.path.realpath(nnetFile)))
    # Something went wrong, so don't write the NNet file
    else:
        print("Could not write NNet file!")


# def export_onnx(path, batch_size, seq_len):
#     print('The model is also exported in ONNX format at {}'.
#           format(os.path.realpath(args.onnx_export)))
#     model.eval()
#     dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
#     hidden = model.init_hidden(batch_size)
#     torch.onnx.export(model, (dummy_input, hidden), path)

##in main.py after convertion to onnx
# # Load ONNX model and convert to TensorFlow format
# model_onnx = onnx.load('mnist_dNet.onnx')
# tf_rep = prepare(model_onnx)
# # Print out tensors and placeholders in model (helpful during inference in TensorFlow)
# print(tf_rep.tensor_dict)