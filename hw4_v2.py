import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers import Dense , Activation , Dropout ,Flatten,concatenate
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
#from keras.optimizers import RMSprop
from keras.initializers import glorot_uniform
#from keras.applications.resnet50 import conv_block, identity_block
#from keras.applications.resnet50 import resnet50
#identity_block, conv_block = resnet50.identity_block, resnet50.conv_block

container = np.load('raw_data2.npz')
images = [container[key] for key in container]
images = np.asarray(images).reshape(12992,64,64,1)

container = np.load('raw_label.npz')
labels = [container[key] for key in container]
labels = np.asarray(labels)

#train_x, val_x, train_y, val_y = train_test_split(images, labels, test_size=0.2 , random_state=42)

batch_size = 64
epochs = 200
kernel_regularizer = l2(1e-5)
activation = 'relu'
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
class_weight = {0: 20.0, 1: 50.0, 2: 20.0, 3: 50.0, 4: 1.2, 5: 1.0, 6: 20.0, 7: 20.0}

#Main CNN model with four Convolution layer & two fully connected layer
def my_model():
	#model = Sequential()
	inputs = Input(shape=(64,64,1))
	# 1st Convolutional Layer
	cov1 = (Conv2D(filters=96, kernel_size=(3,3), strides=(2,2), padding='same',kernel_initializer = glorot_uniform(seed=0)))(inputs)
	cov1 = (BatchNormalization(axis=3))(cov1)
	cov1 = (Activation('relu'))(cov1)
	
	cov2 = (Conv2D(filters=96, kernel_size=(3,3), padding='same',kernel_initializer = glorot_uniform(seed=0)))(cov1)
	cov2 = (BatchNormalization(axis=3))(cov2)
	cov2 = (Activation('relu'))(cov2)
	
	cov3 = (Conv2D(filters=96,  kernel_size=(3,3), padding='same',kernel_initializer = glorot_uniform(seed=0)))(cov2)
	cov3 = (BatchNormalization(axis=3))(cov3)
	cov3 = (Activation('relu'))(cov3)
	# Max Pooling
	max1 = (MaxPooling2D(pool_size=(4,4), strides=(2,2), padding='valid'))(cov3)
	max1 = (Dropout(0.5))(max1)

	# 2nd Convolutional Layer
	cov4 = (Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding='same',kernel_initializer = glorot_uniform(seed=0)))(max1)
	cov4 = (BatchNormalization(axis=3))(cov4)
	cov4 = (Activation('relu'))(cov4)
	# Max Pooling
	max2 = (MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))(cov4)
	max2 = (Dropout(0.5))(max2)

	# 3rd Convolutional Layer
	cov5 = (Conv2D(filters=384, kernel_size=(3,1), strides=(1,1), padding='same',kernel_initializer = glorot_uniform(seed=0)))(max2)
	cov5 = (BatchNormalization(axis=3))(cov5)
	cov5 = (Activation('relu'))(cov5)
	
	cov6 = (Conv2D(filters=384, kernel_size=(1,3), strides=(1,1), padding='same',kernel_initializer = glorot_uniform(seed=0)))(max2)
	cov6 = (BatchNormalization(axis=3))(cov1)
	cov6 = (Activation('relu'))(cov5)
	concatenated1 = concatenate([cov5, cov6])
	# 4th Convolutional Layer
	cov7 = (Conv2D(filters=384, kernel_size=(3,1), strides=(1,1), padding='same',kernel_initializer = glorot_uniform(seed=0)))(concatenated1)
	cov7 = (BatchNormalization(axis=3))(cov7)
	cov7 = (Activation('relu'))(cov7)
	
	cov8 = (Conv2D(filters=384, kernel_size=(1,3), strides=(1,1), padding='same',kernel_initializer = glorot_uniform(seed=0)))(concatenated1)
	cov8 = (BatchNormalization(axis=3))(cov8)
	cov8 = (Activation('relu'))(cov8)
	concatenated2 = concatenate([cov7, cov8])
	
	# 5th Convolutional Layer
	cov9 = (Conv2D(filters=256, kernel_size=(3,1), strides=(1,1), padding='same',kernel_initializer = glorot_uniform(seed=0)))(concatenated2)
	cov9 = (BatchNormalization(axis=3))(cov9)
	cov9 = (Activation('relu'))(cov9)
	
	cov10 = (Conv2D(filters=256, kernel_size=(1,3), strides=(1,1), padding='same',kernel_initializer = glorot_uniform(seed=0)))(concatenated2)
	cov10 = (BatchNormalization(axis=3))(cov10)
	cov10 = (Activation('relu'))(cov10)
	concatenated3 = concatenate([cov9, cov10])
	# Max Pooling
	max3 = (MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))(concatenated3)
	max3 = (Dropout(0.5))(max3)

	# Passing it to a Fully Connected layer
	x = (Flatten())(max3)
	# 1st Fully Connected Layer
	x = (Dense(512))(x)
	x = (BatchNormalization())(x)
	x = (Activation('relu'))(x)
	# Add Dropout to prevent overfitting
	x = (Dropout(0.5))(x)

	# 2nd Fully Connected Layer
	x = (Dense(512))(x)
	x = (BatchNormalization())(x)
	x = (Activation('relu'))(x)
	# Add Dropout
	x = (Dropout(0.5))(x)
	

	predictions = (Dense(8, activation='softmax', kernel_regularizer=kernel_regularizer))(x)
	
	model = Model(inputs=inputs, outputs=predictions)
	
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

	
model = my_model()
print(model.summary())


#Generate augmented data
train_datagen = ImageDataGenerator(rescale=1./255,
	rotation_range=30,
    width_shift_range=0.2,  
    height_shift_range=0.2, 
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow(
    images,
	labels,
    batch_size=batch_size,
    subset='training')

validation_generator = train_datagen.flow(
    images,
	labels,
    batch_size=batch_size,
    subset='validation')


modelF = model.fit_generator(train_generator,
                    steps_per_epoch=163,
                    epochs=epochs,
                    validation_data = validation_generator,
                    callbacks=callbacks_list,verbose = 1,
					validation_steps = 41,
                    class_weight = class_weight)

#print(confusion_matrix(model.predict(val_x).argmax(axis=1), val_y.argmax(axis=1)))
					
acc = modelF.history['acc']
val_acc = modelF.history['val_acc']
loss = modelF.history['loss']
val_loss = modelF.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()