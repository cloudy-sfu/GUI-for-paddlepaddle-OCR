import copy
from character import CharacterOps
from paddle_fluid_initialization import set_config
from processor import sorted_boxes, get_rotate_crop_image, resize_norm_img_rec
import cv2
import numpy as np


character_dict_path = 'inference_model/ppocr_keys_v1.txt'
cls_pretrained_model_path = 'inference_model/angle_cls'
rec_pretrained_model_path = 'inference_model/character_rec'

cls_predictor, cls_input_tensor, cls_output_tensors = set_config(cls_pretrained_model_path)
rec_predictor, rec_input_tensor, rec_output_tensors = set_config(rec_pretrained_model_path)
char_ops_params = {
    'character_type': 'ch',
    'character_dict_path': character_dict_path,
    'loss_type': 'ctc',
    'max_text_length': 25,
    'use_space_char': True
}
char_ops = CharacterOps(char_ops_params)


def recognize_text(image, boxes, text_thresh=0.5, angle_classification_thresh=0.9, batch_num=30):
    img_crop_list = []
    boxes = sorted_boxes(boxes)
    for num_box in range(len(boxes)):
        tmp_box = copy.deepcopy(boxes[num_box])
        img_crop = get_rotate_crop_image(image, tmp_box)
        img_crop_list.append(img_crop)

    img_num = len(img_crop_list)
    # Calculate the aspect ratio of all text bars
    width_list = []
    for img in img_crop_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    # Sorting can speed up the recognition process
    indices = np.argsort(np.array(width_list))

    rec_results = [['', 0.0]] * img_num
    cls_res = [['', 0.0]] * img_num

    for beg_img_no in range(0, img_num, batch_num):
        end_img_no = min(img_num, beg_img_no + batch_num)
        norm_img_batch = []
        max_wh_ratio = 0
        for ino in range(beg_img_no, end_img_no):
            h, w = img_crop_list[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for ino in range(beg_img_no, end_img_no):
            norm_img = resize_norm_img_rec(img_crop_list[indices[ino]], max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)

        norm_img_batch = np.concatenate(norm_img_batch, axis=0)
        norm_img_batch = norm_img_batch.copy()

        # recognize
        rec_input_tensor.copy_from_cpu(norm_img_batch)
        rec_predictor.zero_copy_run()
        rec_idx_batch = rec_output_tensors[0].copy_to_cpu()
        rec_idx_lod = rec_output_tensors[0].lod()[0]
        predict_batch = rec_output_tensors[1].copy_to_cpu()
        predict_lod = rec_output_tensors[1].lod()[0]

        # classify
        cls_input_tensor.copy_from_cpu(norm_img_batch)
        cls_predictor.zero_copy_run()
        prob_out = cls_output_tensors[0].copy_to_cpu()
        label_out = cls_output_tensors[1].copy_to_cpu()
        if len(label_out.shape) != 1:
            prob_out, label_out = label_out, prob_out

        # recognize
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
            # if len(valid_ind) == 0:
            if valid_ind.shape[0] == 0:
                continue
            score = np.mean(probs[valid_ind, ind[valid_ind]])
            rec_results[indices[beg_img_no + rno]] = [preds_text, score]

        # classify
        for rno in range(len(label_out)):
            label_idx = label_out[rno]
            score = prob_out[rno][label_idx]
            label = ['0', '180'][label_idx]
            cls_res[indices[beg_img_no + rno]] = [label, score]
            if '180' in label and score > angle_classification_thresh:
                img_crop_list[indices[beg_img_no + rno]] = cv2.rotate(
                    img_crop_list[indices[beg_img_no + rno]], 1)

    # if the recognized text confidence score is lower than text_thresh, then drop it
    rec_res_final = []
    for index_, res in enumerate(rec_results):
        text_, score = res
        if score >= text_thresh:
            rec_res_final.append({
                'text': text_,
                'confidence': float(score),
                'text_box_position': boxes[index_].astype(np.int).tolist()
            })
    # List[ each_block: Dict["text": str, "confidence": float, "text_box_position": List[4*2] ] ]
    # text_box_position: [LU: [col_id, row_id], RU: ..., RD: ..., LD: ...]
    return rec_res_final
