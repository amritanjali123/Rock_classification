
# coding: utf-8

# In[ ]:


import pickle 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN The model type that we will be using is Sequential. 
#Sequential is the easiest way to build a model in Keras. It allows you to build a model layer by layer.
classifier = Sequential() 

# Step 1 - Convolution  
#Activation is the activation function for the layer. The activation function 
#, or Rectified Linear Activation. This activation function has been proven to work well in neural networks.
classifier.add(Convolution2D(32, 3, 3,input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Flatten serves as a connection between the convolution and dense layers.
# Step 3 - Flattening
classifier.add(Flatten())
#‘Dense’ is the layer type we will use in for our output layer.
#Dense is a standard layer type that is used in many cases for neural networks.
# Step 4 - Full connection   ‘add()’ function to add layers to our model.
classifier.add(Dense(output_dim = 128, activation = 'relu'))
"""The activation is ‘softmax’. Softmax makes the output sum up to 1 so the output can be interpreted as probabilities.
The model will then make its prediction based on which option has the highest probability.
We will use ‘categorical_crossentropy’ for our loss function. This is the most common choice for classification.
A lower score indicates that the model is performing better.
To make things even easier to interpret, we will use the ‘accuracy’ metric to see the accuracy score on the validation set when we train the model."""
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
#classifier.add(Dense(output_dim =3, activation = 'softmax'))
#softmax

# Compiling the CNN
"""The optimizer controls the learning rate. We will be using ‘adam’ as our optmizer. Adam is generally a good optimizer to use for many cases. 
The adam optimizer adjusts the learning rate throughout training."""
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

"""The number of epochs is the number of times the model will cycle through the data.
The more epochs we run, the more the model will improve, up to a certain point. 
After that point, the model will stop improving during each epoch. 
For our model, we will set the number of epochs to 25."""
classifier.fit_generator(training_set,
                         samples_per_epoch =11998,
                         nb_epoch = 3,
                         validation_data = validation_set,
                         nb_val_samples = 2399
                        )
                         


import numpy as np
from keras.preprocessing import image
                        
                         


# In[ ]:


data = open('123.csv', 'r')
eng = open('1234.csv', 'w')
eng.write('Image_File,Class')
eng.write('\n')
l1 = data.readlines()
import numpy as np
count =0
from PIL import Image
import PIL
import numpy as np
for i in l1[1:7534]:
    l2 = i.split(',')
    try:
        
        eng.write(l2[0])
        eng.write(',')

       # print()
        img = Image.open('F:'+'\\'+'madhu'+'\\'+'Convolutional_Neural_Networks'+'\\'+'dataset'+'\\'+'Test'+'\\'+l2[0])
        #img = img.convert('L')
        if img.getdata().mode == "RGBA":
            img = img.convert('RGB')
        img = img.resize((64, 64), PIL.Image.ANTIALIAS)
        arr = np.array(img)

        img = np.reshape(img, (1, 64, 64, 3))
        img = img / 255
        result = classifier.predict(img)
        if result[0][0] > 0.5:
            l2[1]="Small"
         # prediction="Small"
            print(result[0][0])
        else:
            l2[1]="Large"
          #prediction="large"
            print(result[0][0])

        eng.write(l2[1])
        eng.write('\n') 
    except:
        #from IPython.display import Image
        #x=Image('F:'+'\\'+'madhu'+'\\'+'Convolutional_Neural_Networks'+'\\'+'dataset'+'\\'+'Test'+'\\'+l2[0])
        #display(x)
        #coun=
        print(l2[0])
        

    

