# EIP4_Assignment_4

1. Validation accuracy of base model - 
      81.89

2.  Your model definition (model.add... ) with output channel size and receptive field:

mymodel = Sequential()
mymodel.add(SeparableConv2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))  #32  #RF=3
mymodel.add(Activation('relu'))
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(48, 3, 3))   #30  #RF=5
mymodel.add(Activation('relu'))
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(MaxPooling2D(pool_size=(2, 2)))   #15    #RF=7
mymodel.add(Dropout(0.25))

mymodel.add(SeparableConv2D(96, 3, 3, border_mode='same'))   #15   #RF=11
mymodel.add(Activation('relu'))
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(96, 3, 3))    #13   #RF=15
mymodel.add(Activation('relu'))
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(MaxPooling2D(pool_size=(2, 2)))   #6   #RF=17
mymodel.add(Dropout(0.25))

mymodel.add(SeparableConv2D(192, 3, 3, border_mode='same'))   #6   #RF=25
mymodel.add(Activation('relu'))
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(192, 3, 3))   #4    #RF=33
mymodel.add(Activation('relu'))
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(10, 4, 4))   #1      #RF=45
mymodel.add(Activation('relu'))
mymodel.add(BatchNormalization())

mymodel.add(Flatten())
mymodel.add(Activation('softmax'))
mymodel.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.03), metrics=['accuracy'])


3.  50 EPOCH LOGS :

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:14: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:14: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, steps_per_epoch=390, epochs=50)`
  
Epoch 1/50
390/390 [==============================] - 21s 55ms/step - loss: 1.4250 - acc: 0.4920 - val_loss: 2.2883 - val_acc: 0.4764
Epoch 2/50
390/390 [==============================] - 19s 49ms/step - loss: 1.0303 - acc: 0.6364 - val_loss: 1.3509 - val_acc: 0.6223
Epoch 3/50
390/390 [==============================] - 19s 48ms/step - loss: 0.8939 - acc: 0.6849 - val_loss: 1.0775 - val_acc: 0.6705
Epoch 4/50
390/390 [==============================] - 19s 48ms/step - loss: 0.8155 - acc: 0.7147 - val_loss: 0.9604 - val_acc: 0.6949
Epoch 5/50
390/390 [==============================] - 19s 48ms/step - loss: 0.7651 - acc: 0.7314 - val_loss: 0.8792 - val_acc: 0.6992
Epoch 6/50
390/390 [==============================] - 19s 48ms/step - loss: 0.7229 - acc: 0.7466 - val_loss: 0.8254 - val_acc: 0.7292
Epoch 7/50
390/390 [==============================] - 19s 48ms/step - loss: 0.6889 - acc: 0.7586 - val_loss: 0.7479 - val_acc: 0.7522
Epoch 8/50
390/390 [==============================] - 19s 48ms/step - loss: 0.6673 - acc: 0.7638 - val_loss: 0.7366 - val_acc: 0.7600
Epoch 9/50
390/390 [==============================] - 19s 48ms/step - loss: 0.6401 - acc: 0.7754 - val_loss: 0.6745 - val_acc: 0.7792
Epoch 10/50
390/390 [==============================] - 19s 48ms/step - loss: 0.6195 - acc: 0.7827 - val_loss: 1.1915 - val_acc: 0.6572
Epoch 11/50
390/390 [==============================] - 19s 48ms/step - loss: 0.6018 - acc: 0.7881 - val_loss: 0.6945 - val_acc: 0.7697
Epoch 12/50
390/390 [==============================] - 19s 48ms/step - loss: 0.5866 - acc: 0.7930 - val_loss: 0.7171 - val_acc: 0.7669
Epoch 13/50
390/390 [==============================] - 19s 48ms/step - loss: 0.5755 - acc: 0.7989 - val_loss: 1.2524 - val_acc: 0.6329
Epoch 14/50
390/390 [==============================] - 19s 48ms/step - loss: 0.5574 - acc: 0.8037 - val_loss: 0.6481 - val_acc: 0.7891
Epoch 15/50
390/390 [==============================] - 19s 48ms/step - loss: 0.5442 - acc: 0.8085 - val_loss: 0.7830 - val_acc: 0.7420
Epoch 16/50
390/390 [==============================] - 19s 48ms/step - loss: 0.5292 - acc: 0.8156 - val_loss: 0.6341 - val_acc: 0.7935
Epoch 17/50
390/390 [==============================] - 19s 48ms/step - loss: 0.5208 - acc: 0.8159 - val_loss: 0.6789 - val_acc: 0.7826
Epoch 18/50
390/390 [==============================] - 19s 48ms/step - loss: 0.5098 - acc: 0.8195 - val_loss: 0.7623 - val_acc: 0.7484
Epoch 19/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4998 - acc: 0.8253 - val_loss: 0.8420 - val_acc: 0.7316
Epoch 20/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4946 - acc: 0.8262 - val_loss: 0.8017 - val_acc: 0.7382
Epoch 21/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4785 - acc: 0.8322 - val_loss: 0.6954 - val_acc: 0.7713
Epoch 22/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4728 - acc: 0.8341 - val_loss: 0.6361 - val_acc: 0.7924
Epoch 23/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4664 - acc: 0.8354 - val_loss: 0.6155 - val_acc: 0.8003
Epoch 24/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4538 - acc: 0.8417 - val_loss: 0.6909 - val_acc: 0.7809
Epoch 25/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4514 - acc: 0.8398 - val_loss: 0.6040 - val_acc: 0.8039
Epoch 26/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4457 - acc: 0.8425 - val_loss: 0.5710 - val_acc: 0.8127
Epoch 27/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4366 - acc: 0.8467 - val_loss: 0.6710 - val_acc: 0.7783
Epoch 28/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4288 - acc: 0.8492 - val_loss: 0.6591 - val_acc: 0.7853
Epoch 29/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4262 - acc: 0.8493 - val_loss: 0.7894 - val_acc: 0.7554
Epoch 30/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4221 - acc: 0.8509 - val_loss: 0.6156 - val_acc: 0.7975
Epoch 31/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4107 - acc: 0.8558 - val_loss: 0.5812 - val_acc: 0.8085
Epoch 32/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4041 - acc: 0.8571 - val_loss: 0.5892 - val_acc: 0.8101
Epoch 33/50
390/390 [==============================] - 19s 48ms/step - loss: 0.4050 - acc: 0.8557 - val_loss: 0.6079 - val_acc: 0.7971
Epoch 34/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3951 - acc: 0.8604 - val_loss: 0.7041 - val_acc: 0.7749
Epoch 35/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3935 - acc: 0.8609 - val_loss: 0.9111 - val_acc: 0.7185
Epoch 36/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3848 - acc: 0.8642 - val_loss: 0.5938 - val_acc: 0.8083
Epoch 37/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3818 - acc: 0.8655 - val_loss: 0.5556 - val_acc: 0.8276
Epoch 38/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3842 - acc: 0.8631 - val_loss: 0.6925 - val_acc: 0.7803
Epoch 39/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3758 - acc: 0.8658 - val_loss: 0.5528 - val_acc: 0.8246
Epoch 40/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3670 - acc: 0.8712 - val_loss: 0.6211 - val_acc: 0.8107
Epoch 41/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3696 - acc: 0.8691 - val_loss: 0.6422 - val_acc: 0.7973
Epoch 42/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3635 - acc: 0.8713 - val_loss: 0.6006 - val_acc: 0.8064
Epoch 43/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3668 - acc: 0.8711 - val_loss: 0.7162 - val_acc: 0.7845
Epoch 44/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3579 - acc: 0.8740 - val_loss: 0.5926 - val_acc: 0.8097
Epoch 45/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3609 - acc: 0.8709 - val_loss: 0.6134 - val_acc: 0.8086
Epoch 46/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3511 - acc: 0.8746 - val_loss: 0.6024 - val_acc: 0.8138
Epoch 47/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3511 - acc: 0.8758 - val_loss: 0.5759 - val_acc: 0.8176
Epoch 48/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3486 - acc: 0.8761 - val_loss: 0.5787 - val_acc: 0.8191
Epoch 49/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3377 - acc: 0.8799 - val_loss: 0.6408 - val_acc: 0.7989
Epoch 50/50
390/390 [==============================] - 19s 48ms/step - loss: 0.3420 - acc: 0.8797 - val_loss: 0.6261 - val_acc: 0.8073
Model took 942.20 seconds to train

Accuracy on test data is: 80.73   (82.76 in 37th Epoch).
