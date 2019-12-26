import os
import cv2
from math import sin, cos, radians
import numpy as np
import tensorflow as tf

# ↓　★★

path = "./faces/train"
dirs = os.listdir(path)
dirs = [f for f in dirs if os.path.isdir(os.path.join(path, f))]

label_dict = {}
i = 0

for dirname in dirs:
    label_dict[dirname] = i
    i += 1

names = dirs


def get_batch_list(l, batch_size):
    # [1, 2, 3, 4, 5,...] -> [[1, 2, 3], [4, 5,..]]
    return [
        np.asarray(l[_:_ + batch_size]) for _ in range(0, len(l), batch_size)
    ]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def inference(images_placeholder, keep_prob):

    x_image = tf.reshape(images_placeholder, [-1, 32, 32, 3])

    # Convolution layer
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer
    h_pool1 = max_pool_2x2(h_conv1)

    # Convolution layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Pooling layer
    h_pool2 = max_pool_2x2(h_conv2)

    # Full connected layer
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Full connected layer
    W_fc2 = weight_variable([1024, len(label_dict)])
    b_fc2 = bias_variable([len(label_dict)])

    return tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# cv2.cv.CV_FOURCC
def cv_fourcc(c1, c2, c3, c4):
    return (ord(c1) & 255) + ((ord(c2) & 255) << 8) + \
        ((ord(c3) & 255) << 16) + ((ord(c4) & 255) << 24)
# ↑　★★

camera = cv2.VideoCapture(0)
face = cv2.CascadeClassifier("opencv-4.1.1/data/haarcascades/haarcascade_frontalface_alt2.xml")

settings = {
    'scaleFactor': 1.3,
    'minNeighbors': 3,
    'minSize': (50, 50),
    'flags': cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV_HAAR_DO_ROUGH_SEARCH
}


def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result


def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1]*0.4
    y = pos[1] - img.shape[0]*0.4
    newx = x*cos(radians(angle)) + y*sin(radians(angle)) + img.shape[1]*0.4
    newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + img.shape[0]*0.4
    return int(newx), int(newy), pos[2], pos[3]


if __name__ == '__main__':

    # 定数定義
    ESC_KEY = 27  # Escキー
    INTERVAL = 33  # 待ち時間
    FRAME_RATE = 30  # fps

    WINDOW_NAME = "detect"
    # FILE_NAME = "detect.avi"

    DEVICE_ID = 0

    # 分類器の指定
    cascade_file = "opencv-4.1.1/data/haarcascades/haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # 保存ビデオファイルの準備
    # rec = cv2.VideoWriter(FILE_NAME, cv_fourcc('X', 'V', 'I', 'D'), FRAME_RATE, (width, height), True)

    # ウィンドウの準備
    cv2.namedWindow(WINDOW_NAME)

    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 1.5

    print("setup tensorflow")
    x = tf.placeholder('float', shape=[None,
                                       32 * 32 * 3])  # 32 * 32, 3 channels
    keep_prob = tf.placeholder('float')
    y_conv = inference(x, keep_prob)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print("loading model data")
    tf.train.Saver().restore(sess, "model_face/model.ckpt")

    while end_flag:
        img = c_frame

        for angle in [0, -25, 25]:
            rimg = rotate_image(img, angle)
            detected = face.detectMultiScale(rimg, **settings)
            if len(detected):
                detected = [rotate_point(detected[-1], img, -angle)]
                break

        # Make a copy as we don't want to draw on the original image:
        for x, y, w, h in detected[-1:]:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('facedetect', img)

        if cv2.waitKey(5) != -1:
            break

        cv2.destroyWindow("facedetect")
