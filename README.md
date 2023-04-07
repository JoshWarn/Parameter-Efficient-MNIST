# Parameter-Efficient-MNIST
How few parameters do you need to get 99% on MNIST?
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                 
 rescaling (Rescaling)       (None, 28, 28, 1)         0         
                                                                 
 conv2d (Conv2D)             (None, 26, 26, 7)         70        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 7))       0                                                                       
                                                                 
 dropout (Dropout)           (None, 13, 13, 7)         0         
                                                                 
 separable_conv2d (Separable  (None, 11, 11, 10)       143       
 Conv2D)                                                         
                                                                 
 dropout_1 (Dropout)         (None, 11, 11, 10)        0         
                                                                 
 separable_conv2d_1 (Separab  (None, 9, 9, 10)         200       
 leConv2D)                                                       
                                                                 
 dropout_2 (Dropout)         (None, 9, 9, 10)          0         
                                                                 
 separable_conv2d_2 (Separab  (None, 7, 7, 11)         211       
 leConv2D)                                                       
                                                                 
 dropout_3 (Dropout)         (None, 7, 7, 11)          0         
                                                                 
 separable_conv2d_3 (Separab  (None, 5, 5, 12)         243       
 leConv2D)                                                       
                                                                 
 dropout_4 (Dropout)         (None, 5, 5, 12)          0         
                                                                 
 global_average_pooling2d (G  (None, 12)               0         
 lobalAveragePooling2D)                                          
                                                                 
 dense (Dense)               (None, 10)                130       
=================================================================
Total params: 997
Trainable params: 997
Non-trainable params: 0
_________________________________________________________________
