import numpy as np
from paddle_fluid_initialization import set_config
from processor import DBProcessTest, DBPostProcess

pretrained_model_path = 'inference_model/text_detection'

predictor, input_tensor, output_tensors = set_config(pretrained_model_path)
preprocessor = DBProcessTest(params={'max_side_len': 960})
postprocessor = DBPostProcess(
    params={
        'thresh': 0.3,
        'box_thresh': 0.5,
        'max_candidates': 1000,
        'unclip_ratio': 1.6
    })


def order_points_clockwise(pts):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-coordinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates, so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most

    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect


def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


def filter_tag_det_res(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box in dt_boxes:
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new)
    return dt_boxes


def detect_text(original_image):
    im, ratio_list = preprocessor(original_image)
    im = im.copy()
    input_tensor.copy_from_cpu(im)
    predictor.zero_copy_run()
    outputs = []
    for output_tensor_ in output_tensors:
        output = output_tensor_.copy_to_cpu()
        outputs.append(output)
    dt_boxes_list = postprocessor({'maps': outputs[0]}, [ratio_list])
    boxes = filter_tag_det_res(dt_boxes_list[0],
                               original_image.shape)
    res = boxes.astype(np.float32)
    return res
