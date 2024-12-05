[![ML Pipeline](https://github.com/peeyushsinghal/Basic_NN_Training/blob/main/.github/workflows/cnn-test.yml/badge.svg)(https://github.com/peeyushsinghal/Basic_NN_Training/blob/main/.github/workflows/cnn-test.yml)
![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)

# MNIST Digit Classification


A PyTorch-based implementation for MNIST digit classification using a Convolutional Neural Network (CNN).

## Project Structure 
```
├── config.yml # Configuration parameters
├── data.py # Data loading and preprocessing
├── main.py # Main execution script
├── model.py # Neural network architecture
├── train.py # Training logic
├── test.py # Testing and evaluation
├── test_cases.py # Testing cases to check the accuracy of the model etc.
├── README.md # README file
└── metrics.json # Metrics file
```


## Configuration

The project uses a YAML configuration file (`config.yml`) to manage hyperparameters and settings:

### Training Parameters
- Epochs: 20
- Learning Rate: 0.01
- Momentum: 0.9
- Weight Decay: 0.0001
- Batch Size: 128

### Model Architecture
- Initial Channels: 8
- Dropout Rate: 0.1

### Data Augmentation
- Random Rotation: ±7 degrees
- Normalization: Mean=0.1307, Std=0.3081

## Usage

0. Install the dependencies:
```
pip install -r requirements.txt
```

1. Execute the execution script:
```
python main.py
```

## Features

- Modular code structure with separate files for different components
- Configuration-driven architecture using YAML
- Data augmentation with random rotations
- CUDA support for GPU acceleration
- Normalized input data
- Configurable model architecture

## Model Architecture

The CNN model consists of:
- Convolutional layers for feature extraction
- Batch normalization for training stability
- Dropout for regularization
- ReLU activation function
- Adaptive Average Pooling to reduce the dimensions of the output to a 1x1 matrix
```
Net(
  (conv1): Sequential(
    (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.1, inplace=False)
  )
  (conv2): Sequential(
    (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.1, inplace=False)
  )
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Sequential(
    (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.1, inplace=False)
  )
  (conv4): Sequential(
    (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.1, inplace=False)
  )
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (filter): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (conv5): Sequential(
    (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.1, inplace=False)
  )
  (conv6): Conv2d(16, 10, kernel_size=(3, 3), stride=(1, 1), bias=False)
  (gap): AdaptiveAvgPool2d(output_size=1)
)
```
## Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
           Dropout-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             576
              ReLU-6            [-1, 8, 28, 28]               0
       BatchNorm2d-7            [-1, 8, 28, 28]              16
           Dropout-8            [-1, 8, 28, 28]               0
         MaxPool2d-9            [-1, 8, 14, 14]               0
           Conv2d-10           [-1, 16, 14, 14]           1,152
             ReLU-11           [-1, 16, 14, 14]               0
      BatchNorm2d-12           [-1, 16, 14, 14]              32
          Dropout-13           [-1, 16, 14, 14]               0
           Conv2d-14           [-1, 16, 14, 14]           2,304
             ReLU-15           [-1, 16, 14, 14]               0
      BatchNorm2d-16           [-1, 16, 14, 14]              32
          Dropout-17           [-1, 16, 14, 14]               0
        MaxPool2d-18             [-1, 16, 7, 7]               0
           Conv2d-19              [-1, 8, 7, 7]             128
           Conv2d-20             [-1, 16, 5, 5]           1,152
             ReLU-21             [-1, 16, 5, 5]               0
      BatchNorm2d-22             [-1, 16, 5, 5]              32
          Dropout-23             [-1, 16, 5, 5]               0
           Conv2d-24             [-1, 10, 3, 3]           1,440
AdaptiveAvgPool2d-25             [-1, 10, 1, 1]               0
================================================================
Total params: 6,952
Trainable params: 6,952
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.61
Params size (MB): 0.03
Estimated Total Size (MB): 0.64
----------------------------------------------------------------

Number of parameters: 6952
```

## Training Logs
```
Epoch 1/20
loss=0.0870 | accuracy=91.41%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:26<00:00, 17.82it/s]

Test set: Average loss: 0.0968, Accuracy: 9713/10000 (97.13%)


Epoch 2/20
loss=0.0616 | accuracy=97.30%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:26<00:00, 17.65it/s]

Test set: Average loss: 0.0557, Accuracy: 9834/10000 (98.34%)


Epoch 3/20
loss=0.0678 | accuracy=97.79%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:26<00:00, 17.40it/s]

Test set: Average loss: 0.0438, Accuracy: 9867/10000 (98.67%)


Epoch 4/20
loss=0.0689 | accuracy=98.20%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:27<00:00, 16.80it/s]

Test set: Average loss: 0.0427, Accuracy: 9862/10000 (98.62%)


Epoch 5/20
loss=0.0147 | accuracy=98.33%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:29<00:00, 15.68it/s]

Test set: Average loss: 0.0362, Accuracy: 9881/10000 (98.81%)


Epoch 6/20
loss=0.0734 | accuracy=98.45%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:26<00:00, 17.72it/s]

Test set: Average loss: 0.0414, Accuracy: 9864/10000 (98.64%)


Epoch 7/20
loss=0.0120 | accuracy=98.50%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:26<00:00, 17.78it/s]

Test set: Average loss: 0.0362, Accuracy: 9894/10000 (98.94%)


Epoch 8/20
loss=0.0870 | accuracy=98.58%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:26<00:00, 17.93it/s]

Test set: Average loss: 0.0302, Accuracy: 9908/10000 (99.08%)


Epoch 9/20
loss=0.0492 | accuracy=98.64%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:24<00:00, 19.14it/s]

Test set: Average loss: 0.0277, Accuracy: 9916/10000 (99.16%)


Epoch 10/20
loss=0.0393 | accuracy=98.68%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:27<00:00, 17.18it/s]

Test set: Average loss: 0.0272, Accuracy: 9917/10000 (99.17%)


Epoch 11/20
loss=0.0241 | accuracy=98.69%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:28<00:00, 16.73it/s]

Test set: Average loss: 0.0283, Accuracy: 9912/10000 (99.12%)


Epoch 12/20
loss=0.0080 | accuracy=98.84%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:26<00:00, 17.55it/s]

Test set: Average loss: 0.0279, Accuracy: 9913/10000 (99.13%)


Epoch 13/20
loss=0.0149 | accuracy=98.87%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:27<00:00, 16.98it/s]

Test set: Average loss: 0.0302, Accuracy: 9912/10000 (99.12%)


Epoch 14/20
loss=0.0748 | accuracy=98.91%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:27<00:00, 17.27it/s]

Test set: Average loss: 0.0226, Accuracy: 9929/10000 (99.29%)


Epoch 15/20
loss=0.0188 | accuracy=98.83%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:28<00:00, 16.56it/s]

Test set: Average loss: 0.0258, Accuracy: 9921/10000 (99.21%)


Epoch 16/20
loss=0.0393 | accuracy=98.97%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:24<00:00, 19.49it/s]

Test set: Average loss: 0.0240, Accuracy: 9925/10000 (99.25%)


Epoch 17/20
loss=0.0812 | accuracy=98.94%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:24<00:00, 19.26it/s]

Test set: Average loss: 0.0199, Accuracy: 9940/10000 (99.40%)


Epoch 18/20
loss=0.0473 | accuracy=98.97%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:25<00:00, 18.71it/s]

Test set: Average loss: 0.0242, Accuracy: 9923/10000 (99.23%)


Epoch 19/20
loss=0.0488 | accuracy=99.01%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:24<00:00, 19.30it/s]

Test set: Average loss: 0.0234, Accuracy: 9928/10000 (99.28%)


Epoch 20/20
loss=0.0076 | accuracy=98.97%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:24<00:00, 18.86it/s]

Test set: Average loss: 0.0244, Accuracy: 9927/10000 (99.27%)
```


## Data Processing

- Uses MNIST dataset from torchvision
- Applies random rotation for data augmentation
- Normalizes images using precalculated mean and standard deviation
- Supports both CPU and GPU execution
