
# coding: utf-8

# In[1]:


from keras.applications import VGG16
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[2]:


from keras.preprocessing.image import ImageDataGenerator


# In[3]:


for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


# In[4]:


from keras import models
from keras import layers
from keras import optimizers
 
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(19, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()


# In[5]:


train_dir = 'crowdai_train/'
validation_dir = 'crowdai_test/'


# In[6]:


import numpy as np
from random import shuffle


# In[7]:


train_datagen = ImageDataGenerator(
      rescale=1./255,
#       rotation_range=20,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       horizontal_flip=True,
#       fill_mode='nearest'
)
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# Change the batchsize according to your system RAM
train_batchsize = 62
val_batchsize = 10
 
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=True)


# In[8]:


print train_generator.class_indices


# In[9]:


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)


# In[10]:


import cv2


# In[27]:


img3 = cv2.imread('/combo/Shashank/Tom_hl.JPG')
img3 = cv2.resize(img3,(224,224))
img3 = np.reshape(img3,[1,224,224,3])


# In[28]:


disease = model.predict_classes(img3)
print disease


# In[29]:


model.save('plant_disease_final.h5')

