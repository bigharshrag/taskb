from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

#custom parameters
nb_class = 1
hidden_dim = 512

vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, out)

for layer in custom_vgg_model.layers[:19]:
	layer.trainable = False

custom_vgg_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

train_data_dir = 'cropped_data/train'
validation_data_dir = 'cropped_data/test'
nb_train_samples = 419
nb_validation_samples = 100
epochs = 5
batch_size = 16

# training augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

# test augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,  
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='binary')  

test_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

custom_vgg_model.summary()
custom_vgg_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=nb_validation_samples // batch_size)

custom_vgg_model.save_weights('first_try.h5')  