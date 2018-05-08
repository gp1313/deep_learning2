import os
import cv2
import numpy as np

from moviepy.editor import VideoFileClip
import mtcnn_detect_face

import keras.backend as K
import tensorflow as tf

### configurations
clip_loc = "../data/dance_short.mp4"
output_clip_loc = '../tmp/tmp.mp4'

WEIGHTS_PATH = './mtcnn_weights'

detect_threshold = 0.7

### Coding
class VideoRunTime:
    def __init__(self):
        self.frames = 0
        self.prev_x0 = self.prev_y0 = self.prev_x1 = self.prev_y1= None

    def set_prev_coord(self, x0, x1, y0, y1):
        self.prev_x0 = x0
        self.prev_x1 = x1
        self.prev_y1 = y1
        self.prev_y0 = y0


vid_status = VideoRunTime()

#  Change working directory
exec_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(exec_dir)

def process_video(input_img):
    global vid_status
    image = input_img
    faces = get_faces_bbox(image)  # faces: face bbox coord

    most_conf_face = None
    # Non-max suppress.
    if len(faces) > 0:
        most_conf_face, faces = remove_overlaps(faces)

    best_conf_score = 0

    if most_conf_face:
        x0, y1, x1, y0, conf_score = most_conf_face
        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y1 = int(y1)

        # (image, top left (y, x), bottom right (y, x), red, thickness)
        cv2.rectangle(input_img, (y0, x0), (y1, x1), (255, 0, 0), 3)
        vid_status.set_prev_coord(x0, x1, y0, y1)

    vid_status.frames += 1

    return input_img


def get_faces_bbox(image):
    minsize = 20                              # minimum size of face
    threshold = [0.6, 0.7, detect_threshold]  # three steps's threshold
    factor = 0.709                            # scale factor
    video_scaling_factor = 1
    resized_image = image

    if is_higher_than_1080p(image):
        video_scaling_factor = 4
        resized_image = cv2.resize(image,
                                   (image.shape[1] // video_scaling_factor,
                                    image.shape[0] // video_scaling_factor))
    elif is_higher_than_720p(image):
        video_scaling_factor = 3
        resized_image = cv2.resize(image,
                                   (image.shape[1] // video_scaling_factor,
                                    image.shape[0] // video_scaling_factor))
    elif is_higher_than_480p(image):
        video_scaling_factor = 2
        resized_image = cv2.resize(image,
                                   (image.shape[1] // video_scaling_factor,
                                    image.shape[0] // video_scaling_factor))

    faces, pnts = mtcnn_detect_face.detect_face(resized_image, minsize, pnet, rnet, onet, threshold, factor)
    faces = process_mtcnn_bbox(faces, image.shape)
    faces = calibrate_coord(faces, video_scaling_factor)

    return faces


def is_higher_than_480p(x):
    return (x.shape[0] * x.shape[1]) >= (858 * 480)

def is_higher_than_720p(x):
    return (x.shape[0] * x.shape[1]) >= (1280 * 720)

def is_higher_than_1080p(x):
    return (x.shape[0] * x.shape[1]) >= (1920 * 1080)

def process_mtcnn_bbox(bboxes, im_shape):
    """
    Convert MTCNN boundary boxes to a square bboxes in (x0, y1, x1, y0)
    :param bboxes: bboxes from MTCNN (y0, x0, y1, x1)
    :param im_shape: image shape
    :return: square bboxes in (x0, y1, x1, y0)
    """
    for i, bbox in enumerate(bboxes):
        y0, x0, y1, x1 = bboxes[i, 0:4]
        w = int(y1 - y0)
        h = int(x1 - x0)
        radius = (w + h) // 4
        center = (int((x1 + x0) / 2), int((y1 + y0) / 2))
        new_x0 = np.max([0, (center[0] - radius)])
        new_x1 = np.min([im_shape[0], (center[0] + radius)])
        new_y0 = np.max([0, (center[1] - radius)])
        new_y1 = np.min([im_shape[1], (center[1] + radius)])
        bboxes[i, 0:4] = new_x0, new_y1, new_x1, new_y0
    return bboxes


def calibrate_coord(faces, video_scaling_factor):
    """
    Scale the bbox into the original resolution before the scale down.
    :param faces: bboxes
    :param video_scaling_factor: how much it was scale down.
    :return: bboxes after scale up.
    """
    return [[x0 * video_scaling_factor, y1 * video_scaling_factor,
             x1 * video_scaling_factor, y0 * video_scaling_factor, prob]
                for x0, y1, x1, y0, prob in faces]

def remove_overlaps(faces):
    main_face = get_most_conf_face(faces)
    main_face_bbox = main_face[0]
    result_faces = [(x0, y1, x1, y0, conf_score)
                                  for x0, y1, x1, y0, conf_score in faces
                                     if not is_overlap(main_face_bbox, (x0, y1, x1, y0))]
    return main_face_bbox, main_face + result_faces

def get_most_conf_face(faces):
    """
    Return the bbox with the highest confidence score
    :param faces: bboxes
    :return: A one element list containing the highest confidence bbox
    """
    conf_face = max(faces, key = lambda face: face[4])
    return [conf_face]

def is_overlap(box1, box2):
    overlap_x0 = np.max([box1[0], box2[0]]).astype(np.float32)
    overlap_y1 = np.min([box1[1], box2[1]]).astype(np.float32)
    overlap_x1 = np.min([box1[2], box2[2]]).astype(np.float32)
    overlap_y0 = np.max([box1[3], box2[3]]).astype(np.float32)
    area_iou = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    area_box1 = (box1[2] - box1[0]) * (box1[1] - box1[3])
    return (area_iou / area_box1) >= 0.2


def create_mtcnn(sess, model_path):
    with tf.variable_scope('pnet'):
        data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
        pnet = mtcnn_detect_face.PNet({'data':data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet'):
        data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
        rnet = mtcnn_detect_face.RNet({'data':data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet'):
        data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
        onet = mtcnn_detect_face.ONet({'data':data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)
    return pnet, rnet, onet

sess = K.get_session()
with sess.as_default():
    pnet, rnet, onet = create_mtcnn(sess, WEIGHTS_PATH)

pnet = K.function([pnet.layers['data']], [pnet.layers['conv4-2'], pnet.layers['prob1']])
rnet = K.function([rnet.layers['data']], [rnet.layers['conv5-2'], rnet.layers['prob1']])
onet = K.function([onet.layers['data']], [onet.layers['conv6-2'], onet.layers['conv6-3'], onet.layers['prob1']])

bbox_moving_avg_coef = 0.65

clip = VideoFileClip(clip_loc)

# A get around for https://github.com/Zulko/moviepy/issues/682
# using https://github.com/Zulko/moviepy/issues/586
if clip.rotation == 90:
    clip = clip.resize(clip.size[::-1])       # 90 degree rotate
    clip.rotation = 0

clip = clip.fl_image(process_video)           # Call process_video to swap face for every frame in the video
clip.write_videofile(output_clip_loc, audio=False)

