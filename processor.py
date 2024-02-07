import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import math


class DBProcessTest:
    """
    DB pre-process for Test mode
    """

    def __init__(self, params):
        # super(DBProcessTest, self).__init__()
        self.resize_type = 0
        if 'test_image_shape' in params:
            self.image_shape = params['test_image_shape']
            self.resize_type = 1
        if 'max_side_len' in params:
            self.max_side_len = params['max_side_len']
        else:
            self.max_side_len = 2400

    def resize_image_type0(self, im):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        max_side_len = self.max_side_len
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            if resize_h > resize_w:
                ratio = float(max_side_len) / resize_h
            else:
                ratio = float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        if resize_h % 32 == 0:
            resize_h = resize_h
        elif resize_h // 32 <= 1:
            resize_h = 32
        else:
            resize_h = (resize_h // 32 - 1) * 32
        if resize_w % 32 == 0:
            resize_w = resize_w
        elif resize_w // 32 <= 1:
            resize_w = 32
        else:
            resize_w = (resize_w // 32 - 1) * 32
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            im = cv2.resize(im, (int(resize_w), int(resize_h)))
        except Exception as e:
            raise Exception(f'Resizing error. {e}. Image shape: {im.shape}, '
                            f'resized width: {resize_w}, resized height: {resize_h}')
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image_type1(self, im):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = im.shape[:2]  # (h, w, c)
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        return im, (ratio_h, ratio_w)

    def normalize(self, im):
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im[:, :, 0] -= img_mean[0]
        im[:, :, 1] -= img_mean[1]
        im[:, :, 2] -= img_mean[2]
        im[:, :, 0] /= img_std[0]
        im[:, :, 1] /= img_std[1]
        im[:, :, 2] /= img_std[2]
        channel_swap = (2, 0, 1)
        im = im.transpose(channel_swap)
        return im

    def __call__(self, im):
        if self.resize_type == 0:
            im, (ratio_h, ratio_w) = self.resize_image_type0(im)
        else:
            im, (ratio_h, ratio_w) = self.resize_image_type1(im)
        im = self.normalize(im)
        im = im[np.newaxis, :]
        return [im, (ratio_h, ratio_w)]


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, params):
        self.thresh = params['thresh']
        self.box_thresh = params['box_thresh']
        self.max_candidates = params['max_candidates']
        self.unclip_ratio = params['unclip_ratio']
        self.min_size = 3
        self.dilation_kernel = np.array([[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, ratio_list):
        pred = outs_dict['maps']

        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = pred.shape[-2:]

            mask = cv2.dilate(np.array(segmentation[batch_index]).astype(np.uint8), self.dilation_kernel)
            tmp_boxes, tmp_scores = self.boxes_from_bitmap(pred[batch_index], mask, width, height)

            boxes = []
            for k in range(len(tmp_boxes)):
                if tmp_scores[k] > self.box_thresh:
                    boxes.append(tmp_boxes[k])
            if len(boxes) > 0:
                boxes = np.array(boxes)

                ratio_h, ratio_w = ratio_list[batch_index]
                boxes[:, :, 0] = boxes[:, :, 0] / ratio_w
                boxes[:, :, 1] = boxes[:, :, 1] / ratio_h

            boxes_batch.append(boxes)
        return boxes_batch


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes_ = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes_)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def resize_norm_img_cls(img):
    cls_image_shape = [3, 48, 192]
    img_c, img_h, img_w = cls_image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / h
    if math.ceil(img_h * ratio) > img_w:
        resized_w = img_w
    else:
        resized_w = math.ceil(img_h * ratio)
    resized_image = cv2.resize(img, (resized_w, img_h))
    resized_image = resized_image.astype('float32')
    if cls_image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def resize_norm_img_rec(img, max_wh_ratio):
    img_c, img_h, img_w = [3, 32, 320]
    assert img_c == img.shape[2]
    img_w = int((32 * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / h
    if math.ceil(img_h * ratio) > img_w:
        resized_w = img_w
    else:
        resized_w = math.ceil(img_h * ratio)
    resized_image = (cv2.resize(img, (resized_w, img_h)).astype('float32').transpose((2, 0, 1)) / 255 - 0.5) / 0.5
    padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im
