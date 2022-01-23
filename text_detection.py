from processor import DBProcessTest, DBPostProcess, draw_boxes, get_image_ext
from paddle.fluid.core_avx import create_paddle_predictor, AnalysisConfig
import numpy as np
import cv2
import os
from PIL import Image
import time


model_file_path = 'inference_model/text_detection/model'
params_file_path = 'inference_model/text_detection/params'

config = AnalysisConfig(model_file_path, params_file_path)
config.disable_gpu()
config.set_cpu_math_library_num_threads(6)
config.disable_glog_info()

# use zero copy
config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
config.switch_use_feed_fetch_ops(False)
predictor = create_paddle_predictor(config)
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_tensor(input_names[0])
output_names = predictor.get_output_names()
output_tensors = []
for output_name in output_names:
    output_tensor = predictor.get_output_tensor(output_name)
    output_tensors.append(output_tensor)


def order_points_clockwise(pts):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

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


def detect_text(
        images=None,
        # paths=[],
        # use_gpu=False,
        output_dir='detection_result',
        visualization=False,
        # box_thresh=0.5
):
    if not images:
        "There is not any image to be predicted. Please check the input data."
    predicted_data = images
    # assert predicted_data != [], "There is not any image to be predicted. Please check the input data."
    preprocessor = DBProcessTest(params={'max_side_len': 960})
    postprocessor = DBPostProcess(
        params={
            'thresh': 0.3,
            'box_thresh': 0.5,
            'max_candidates': 1000,
            'unclip_ratio': 1.6
        })

    all_imgs = []
    all_ratios = []
    all_results = []
    for original_image in predicted_data:
        # ori_im = original_image.copy()
        im, ratio_list = preprocessor(original_image)
        res = {'save_path': ''}
        if im is None:
            res['data'] = []
        else:
            im = im.copy()
            input_tensor.copy_from_cpu(im)
            predictor.zero_copy_run()
            outputs = []
            for output_tensor_ in output_tensors:
                output = output_tensor_.copy_to_cpu()
                outputs.append(output)
            outs_dict = {}
            outs_dict['maps'] = outputs[0]
            dt_boxes_list = postprocessor(outs_dict, [ratio_list])
            boxes = filter_tag_det_res(dt_boxes_list[0],
                                       original_image.shape)
            res['data'] = boxes.astype(np.int).tolist()

            all_imgs.append(im)
            all_ratios.append(ratio_list)
            if visualization:
                img = Image.fromarray(
                    cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                draw_img = draw_boxes(img, boxes)
                draw_img = np.array(draw_img)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                ext = get_image_ext(original_image)
                saved_name = 'ndarray_{}{}'.format(time.time(), ext)
                cv2.imwrite(
                    os.path.join(output_dir, saved_name),
                    draw_img[:, :, ::-1])
                res['save_path'] = os.path.join(output_dir, saved_name)

        all_results.append(res)
    return all_results


if __name__ == '__main__':
    all_results = detect_text(images=[cv2.imread('test1.png')])
