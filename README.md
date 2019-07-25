# Tool pytorch to nnet
Simple tool to convert a pytorch model .pt to reluplex model .nnet

## How it works

First of all the function converts the .pt model to .onnx format. Then the ONNX model will be converted to .nnet. 
 
## How to use
Import the convertes.py and nsert the function below into the main file of model to convert it to .nnet format.

```
from convertes import *
pt_2_nnet(destination_path_onnx, destination_path_nnet, model,device, batch_size, seq_len)
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
