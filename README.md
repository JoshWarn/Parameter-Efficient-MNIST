# Parameter-Efficient-MNIST
How many parameters are needed to get 99% of MNIST?

Well, 697 parameters and 5 convolution layers later, we have an upper limit!

Inspired by https://github.com/ruslangrimov/mnist-minimal-model

Components optimized:
- Activation function
- Kernel initialization
- Layer count and number of kernels per layer
- Kernel filter sizes/shapes
- Dropout %
- Optimizer (Type, decay, lrs)
- LR Scheduling
- Augmentation (none=best)
- Probably several other things I'm forgetting.
~700 697-parameter models trained (305 plotted).
https://github.com/ThomasWarn contributed ~35 models.
![image](https://github.com/JoshWarn/Parameter-Efficient-MNIST/assets/70070682/ee4dd32c-e995-4345-94f2-ff1583462a95)
![image](https://github.com/JoshWarn/Parameter-Efficient-MNIST/assets/70070682/4062b7e2-6a6d-484d-b0ff-d6e3fa0c1a1f)

