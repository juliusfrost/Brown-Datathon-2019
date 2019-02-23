import numpy as np
from numpy import load

# Load fruit fly data for training.
data = load('fruit_fly_volumes.npz')
train_volume = np.expand_dims(data['volume'], axis=-1)
train_label = np.expand_dims(data['label'], axis=-1)

# Load mouse data for evaluation.
data = load('mouse_volumes.npz')
test_volume = np.expand_dims(data['volume'], axis=-1)
test_label = np.expand_dims(data['label'], axis=-1)




from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam

# Define a U-Net using Keras.
# Code modified from https://github.com/zhixuhao/unet
def small_unet(input_size=(1248, 1248, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)

    up4 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop3))
    merge4 = concatenate([drop2, up4], axis=3)
    conv4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)
    conv4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    up5 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv1, up5], axis=3)
    conv5 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv6 = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(input=inputs, output=conv6)

    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])

    return model




# Train and evaluate.
model = small_unet()
model.fit(x=train_volume, y=train_label, batch_size=20, epochs=1)
print(model.evaluate(x=test_volume, y=test_label))