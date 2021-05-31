import string
import random

import numpy as np
import cv2

charset = string.ascii_lowercase + string.ascii_uppercase + '0123456789'
chargen = np.vectorize(lambda x: ''.join(random.choices(charset, k=x)))


def f(img_bytes: bytes):
    img = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    h, w, *_ = img.shape
    n = 15
    ws = np.random.lognormal(np.log(w / 30), w / 300, n).clip(w / 50, w).astype(int)
    hs = np.random.lognormal(np.log(h / 40), w / 1200, n).clip(h / 50, h).astype(int)
    xs = np.random.randint(0, w - ws + 1, n)
    ys = np.random.randint(0, h - hs + 1, n)
    tls = np.random.lognormal(np.log(10), 0.25, n).astype(int)
    ts = chargen(tls)
    return list(zip(xs, ys, ws, hs, ts))
