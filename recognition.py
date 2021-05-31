from typing import *

import numpy as np
import cv2
from tensorflow import keras

# __all__ = []

emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]


def cnn_print_digit(d):
    print(d.shape)
    for x in range(28):
        s = ""
        for y in range(28):
            s += "{0:.1f} ".format(d[28 * y + x])
        print(s)


def cnn_print_digit_2d(d):
    print(d.shape)
    for y in range(d.shape[0]):
        s = ""
        for x in range(d.shape[1]):
            s += "{0:.1f} ".format(d[x][y])
        print(s)


def predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    # result = model.predict_classes([img_arr])
    result = np.argmax(model.predict(img_arr), axis=-1)
    return chr(emnist_labels[result[0]])


def predict_imgs(model, img_arr):
    img_arr = 1 - np.array(img_arr) / 255.0
    img_arr = np.rot90(img_arr, 3, axes=(1, 2))
    img_arr = np.flip(img_arr, axis=2)
    img_arr = img_arr[:, :28, :28, None]

    result = np.argmax(model.predict(img_arr), axis=-1)
    return list(map(chr, np.array(emnist_labels)[result]))


def letters_extract(img, out_size=28):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, y, w, h, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    # cv2.imshow("Input", img)
    # cv2.imshow("Enlarged", img_erode)
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)

    return letters


def bytes_to_img(img_bytes: bytes):
    return cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def img_predict_letters(model, img):
    letters = letters_extract(img)
    if letters:
        chrs = predict_imgs(model, [let[4] for let in letters])
        for i in range(len(letters)):
            letters[i] = (*letters[i][:4], chrs[i])
    return letters


def img_to_str(model: Any, img):
    letters = letters_extract(img)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i + 1][0] - letters[i][0] - letters[i][2] if i < len(letters) - 1 else 0
        s_out += predict_img(model, letters[i][4])
        if dn > letters[i][1] / 4:
            s_out += ' '
    return s_out


if __name__ == "__main__":
    print("Loading model")
    model = keras.models.load_model('text_writer_model.h5')
    print("Recognition")
    # s_out = img_predict_letters(model, cv2.imread("training/hello.png"))
    # s_out = img_to_str(model, cv2.imread("training/hello.png"))
    s_out = img_predict_letters(model, cv2.imread("test.png"))
    # s_out = img_to_str(model, cv2.imread("training/hello.png"))
    print(s_out)
