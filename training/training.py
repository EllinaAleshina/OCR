import time

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
import idx2numpy

from ..recognition import emnist_labels, predict_img


# Dataset:
# https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip


def text_writer_model():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    return model


def train_model(model):
    t_start = time.time()

    path = "D:\Alexey\Python\TextWriter\gzip"
    X_train = idx2numpy.convert_from_file(path + '\emnist-byclass-train-images-idx3-ubyte')
    y_train = idx2numpy.convert_from_file(path + '\emnist-byclass-train-labels-idx1-ubyte')

    X_test = idx2numpy.convert_from_file(path + '\emnist-byclass-test-images-idx3-ubyte')
    y_test = idx2numpy.convert_from_file(path + '\emnist-byclass-test-labels-idx1-ubyte')

    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(emnist_labels))

    # Test:
    k = 10
    X_train = X_train[:X_train.shape[0] // k]
    y_train = y_train[:y_train.shape[0] // k]
    X_test = X_test[:X_test.shape[0] // k]
    y_test = y_test[:y_test.shape[0] // k]

    # Normalize
    X_train = X_train.astype(np.float32)
    X_train /= 255.0
    X_test = X_test.astype(np.float32)
    X_test /= 255.0

    x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
    y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))

    # Set a learning rate reduction
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5,
                                                                min_lr=0.00001)

    model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction],
              batch_size=64, epochs=30)
    print("Training done, dT:", time.time() - t_start)


def predict(model, image_file):
    img = keras.preprocessing.image.load_img(image_file, target_size=(28, 28), color_mode='grayscale')
    predict_img(model, img)


if __name__ == "__main__":
    model = text_writer_model()
    train_model(model)
    model.save('../text_writer_model.h5')
