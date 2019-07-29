# Tool pytorch to nnet
Simple tool to convert a pytorch model .pt to reluplex model .nnet

## How it works

First of all the function converts the .pt model to .onnx format. Then the ONNX model will be converted to .nnet. 
 
## How to use
Import the convertes.py and insert the function below in the main file of model to convert it to .nnet format.
input parameter is a tensor.
i.e input = torch.Tensor(64 ,1, 28, 28)
```
from convertes import *
pt_2_nnet(destination_path_onnx, destination_path_nnet, model, input)
```

## Modules installed
```
torch.onnx
onnx
```
## Op_type supported

* MatMul
* Add
* ReLU

This supports three types of nodes: MatMul, Add, and Relu
The .nnet file format specifies only feedforward fully-connected Relu networks, so these operations are sufficient to specify nnet networks. If the onnx model uses other operations, the convertion will fail.
