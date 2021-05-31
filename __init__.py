import os

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
# Debug messages
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow import keras

from .recognition import bytes_to_img, img_predict_letters

__all__ = ["bytes_predict_letters", "img_predict_letters"]

model = keras.models.load_model(os.path.join(os.path.dirname(__file__), "text_writer_model.h5"))


def bytes_predict_letters(img_bytes: bytes):
    return img_predict_letters(model, bytes_to_img(img_bytes))
