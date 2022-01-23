import copy
import math
import os

import cv2
import numpy as np
from paddle.fluid.core_avx import create_paddle_predictor, AnalysisConfig

from character import CharacterOps
from text_detection import detect_text
from utils import sorted_boxes

# cls_pretrained_model_path = 'C:\\Users\\cloudy\\.paddlehub\\modules\\chinese_ocr_db_crnn_mobile\\inference_model' \
#                             '\\angle_cls'
# rec_pretrained_model_path = 'C:\\Users\\cloudy\\.paddlehub\\modules\\chinese_ocr_db_crnn_mobile\\inference_model' \
#                             '\\character_rec'
# character_dict_path = 'C:\\Users\\cloudy\\.paddlehub\\modules\\chinese_ocr_db_crnn_mobile\\assets\\ppocr_keys_v1.txt'
character_dict_path = 'inference_model/ppocr_keys_v1.txt'
cls_pretrained_model_path = 'inference_model/angle_cls'
rec_pretrained_model_path = 'inference_model/character_rec'


def _set_config(pretrained_model_path):
    """
    predictor config path
    """
    model_file_path = os.path.join(pretrained_model_path, 'model')
    params_file_path = os.path.join(pretrained_model_path, 'params')

    config = AnalysisConfig(model_file_path, params_file_path)
    try:
        _places = os.environ["CUDA_VISIBLE_DEVICES"]
        int(_places[0])
        use_gpu = True
    except:
        use_gpu = False

    if use_gpu:
        config.enable_use_gpu(8000, 0)
    else:
        config.disable_gpu()

    config.disable_glog_info()
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

    return predictor, input_tensor, output_tensors


cls_predictor, cls_input_tensor, cls_output_tensors = _set_config(
    cls_pretrained_model_path)
rec_predictor, rec_input_tensor, rec_output_tensors = _set_config(
    rec_pretrained_model_path)


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
    imgC, imgH, imgW = cls_image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if cls_image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def _classify_text(image_list, angle_classification_thresh=0.9):
    img_list = copy.deepcopy(image_list)
    img_num = len(img_list)
    # Calculate the aspect ratio of all text bars
    width_list = []
    for img in img_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    # Sorting can speed up the cls process
    indices = np.argsort(np.array(width_list))

    cls_res = [['', 0.0]] * img_num
    batch_num = 30
    for beg_img_no in range(0, img_num, batch_num):
        end_img_no = min(img_num, beg_img_no + batch_num)
        norm_img_batch = []
        max_wh_ratio = 0
        for ino in range(beg_img_no, end_img_no):
            h, w = img_list[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for ino in range(beg_img_no, end_img_no):
            norm_img = resize_norm_img_cls(img_list[indices[ino]])
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)
        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()

        cls_input_tensor.copy_from_cpu(norm_img_batch)
        cls_predictor.zero_copy_run()

        prob_out = cls_output_tensors[0].copy_to_cpu()
        label_out = cls_output_tensors[1].copy_to_cpu()
        if len(label_out.shape) != 1:
            prob_out, label_out = label_out, prob_out
        label_list = ['0', '180']
        for rno in range(len(label_out)):
            label_idx = label_out[rno]
            score = prob_out[rno][label_idx]
            label = label_list[label_idx]
            cls_res[indices[beg_img_no + rno]] = [label, score]
            if '180' in label and score > angle_classification_thresh:
                img_list[indices[beg_img_no + rno]] = cv2.rotate(img_list[indices[beg_img_no + rno]], 1)
    return img_list, cls_res


def resize_norm_img_rec(img, max_wh_ratio):
    imgC, imgH, imgW = [3, 32, 320]
    assert imgC == img.shape[2]
    imgW = int((32 * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


char_ops_params = {
    'character_type': 'ch',
    'character_dict_path': character_dict_path,
    'loss_type': 'ctc',
    'max_text_length': 25,
    'use_space_char': True
}
char_ops = CharacterOps(char_ops_params)


def _recognize_text(img_list):
    img_num = len(img_list)
    # Calculate the aspect ratio of all text bars
    width_list = []
    for img in img_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    # Sorting can speed up the recognition process
    indices = np.argsort(np.array(width_list))

    rec_res = [['', 0.0]] * img_num
    batch_num = 30
    for beg_img_no in range(0, img_num, batch_num):
        end_img_no = min(img_num, beg_img_no + batch_num)
        norm_img_batch = []
        max_wh_ratio = 0
        for ino in range(beg_img_no, end_img_no):
            h, w = img_list[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for ino in range(beg_img_no, end_img_no):
            norm_img = resize_norm_img_rec(img_list[indices[ino]], max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)

        norm_img_batch = np.concatenate(norm_img_batch, axis=0)
        norm_img_batch = norm_img_batch.copy()

        rec_input_tensor.copy_from_cpu(norm_img_batch)
        rec_predictor.zero_copy_run()

        rec_idx_batch = rec_output_tensors[0].copy_to_cpu()
        rec_idx_lod = rec_output_tensors[0].lod()[0]
        predict_batch = rec_output_tensors[1].copy_to_cpu()
        predict_lod = rec_output_tensors[1].lod()[0]
        for rno in range(len(rec_idx_lod) - 1):
            beg = rec_idx_lod[rno]
            end = rec_idx_lod[rno + 1]
            rec_idx_tmp = rec_idx_batch[beg:end, 0]
            preds_text = char_ops.decode(rec_idx_tmp)
            beg = predict_lod[rno]
            end = predict_lod[rno + 1]
            probs = predict_batch[beg:end, :]
            ind = np.argmax(probs, axis=1)
            blank = probs.shape[1]
            valid_ind = np.where(ind != (blank - 1))[0]
            if len(valid_ind) == 0:
                continue
            score = np.mean(probs[valid_ind, ind[valid_ind]])
            # rec_res.append([preds_text, score])
            rec_res[indices[beg_img_no + rno]] = [preds_text, score]

    return rec_res


def recognize_text(
        images=None,
        paths=[],
        use_gpu=False,
        output_dir='ocr_result',
        visualization=False,
        box_thresh=0.5,
        text_thresh=0.5,
        angle_classification_thresh=0.9
):
    if not images:
        "There is not any image to be predicted. Please check the input data."
    predicted_data = images
    detection_results = detect_text(images=predicted_data)
    boxes = [np.array(item['data']).astype(np.float32) for item in detection_results]
    all_results = []
    for index, img_boxes in enumerate(boxes):
        original_image = predicted_data[index].copy()
        result = {'save_path': ''}
        if img_boxes.size == 0:
            result['data'] = []
        else:
            img_crop_list = []
            boxes = sorted_boxes(img_boxes)
            for num_box in range(len(boxes)):
                tmp_box = copy.deepcopy(boxes[num_box])
                img_crop = get_rotate_crop_image(original_image, tmp_box)
                img_crop_list.append(img_crop)
            img_crop_list, angle_list = _classify_text(
                img_crop_list, angle_classification_thresh=angle_classification_thresh)
            rec_results = _recognize_text(img_crop_list)

            # if the recognized text confidence score is lower than text_thresh, then drop it
            rec_res_final = []
            for index, res in enumerate(rec_results):
                text, score = res
                if score >= text_thresh:
                    rec_res_final.append({
                        'text': text,
                        'confidence': float(score),
                        'text_box_position': boxes[index].astype(np.int).tolist()
                    })
            result['data'] = rec_res_final
        all_results.append(result)
    return all_results


if __name__ == '__main__':
    text = recognize_text(images=[cv2.imread('test1.png')])
