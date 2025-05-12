# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import importlib.util
import json
import logging
import os
import shutil
import sys
import tarfile
from argparse import Namespace
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

from paddleocr.ppocr.utils.logging import get_logger
from paddleocr.ppocr.utils.utility import (check_and_read, get_image_file_list,
                                           alpha_to_color, binarize_img)
from paddleocr.tools.infer import predict_system
from paddleocr.tools.infer.utility import draw_ocr, str2bool
from paddleocr.tools.infer.utility import init_args


def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


logger = get_logger()
SUPPORT_DET_MODEL = ['DB']
VERSION = '2.7.0.3'
SUPPORT_REC_MODEL = ['CRNN', 'SVTR_LCNet']
BASE_DIR = os.path.abspath("./inference_models")
os.makedirs(BASE_DIR, exist_ok=True)
DEFAULT_OCR_MODEL_VERSION = 'PP-OCRv4'
SUPPORT_OCR_MODEL_VERSION = ['PP-OCR', 'PP-OCRv2', 'PP-OCRv3', 'PP-OCRv4']
DEFAULT_STRUCTURE_MODEL_VERSION = 'PP-StructureV2'
SUPPORT_STRUCTURE_MODEL_VERSION = ['PP-Structure', 'PP-StructureV2']
with open("./paddleocr/model_urls.json", "r") as f:
    MODEL_URLS = json.load(f)


def parse_args():
    ns = Namespace(
        use_gpu=True, use_xpu=False, use_npu=False, ir_optim=True, use_tensorrt=False,
        min_subgraph_size=15, precision='fp32', gpu_mem=500, gpu_id=0, image_dir=None,
        page_num=0, det_algorithm='DB', det_model_dir=None, det_limit_side_len=960,
        det_limit_type='max', det_box_type='quad', det_db_thresh=0.3,
        det_db_box_thresh=0.6, det_db_unclip_ratio=1.5, max_batch_size=10,
        use_dilation=False, det_db_score_mode='fast', det_east_score_thresh=0.8,
        det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_sast_score_thresh=0.5,
        det_sast_nms_thresh=0.2, det_pse_thresh=0, det_pse_box_thresh=0.85,
        det_pse_min_area=16, det_pse_scale=1, scales=[8, 16, 32], alpha=1.0,
        beta=1.0, fourier_degree=5, rec_algorithm='SVTR_LCNet', rec_model_dir=None,
        rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_batch_num=6,
        max_text_length=25, rec_char_dict_path=None, use_space_char=True,
        vis_font_path='./doc/fonts/simfang.ttf', drop_score=0.5, e2e_algorithm='PGNet',
        e2e_model_dir=None, e2e_limit_side_len=768, e2e_limit_type='max',
        e2e_pgnet_score_thresh=0.5,
        e2e_char_dict_path='./paddleocr/ppocr/utils/ic15_dict.txt',
        e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode='fast', use_angle_cls=False,
        cls_model_dir=None, cls_image_shape='3, 48, 192', label_list=['0', '180'],
        cls_batch_num=6, cls_thresh=0.9, enable_mkldnn=False, cpu_threads=10,
        use_pdserving=False, warmup=False, sr_model_dir=None,
        sr_image_shape='3, 32, 128', sr_batch_num=1,
        draw_img_save_dir='./inference_results', save_crop_res=False,
        crop_res_save_dir='./output', use_mp=False, total_process_num=1, process_id=0,
        benchmark=False, save_log_path='./log_output/', show_log=True, use_onnx=False,
        lang='ch', det=True, rec=True, type='ocr', ocr_version='PP-OCRv4',
        structure_version='PP-StructureV2',
    )
    return ns


def get_model_config(type_, version, model_type, lang):
    if type_ == 'OCR':
        default_model_version = DEFAULT_OCR_MODEL_VERSION
    elif type_ == 'STRUCTURE':
        default_model_version = DEFAULT_STRUCTURE_MODEL_VERSION
    else:
        raise NotImplementedError(f'Function {type_} is not implemented. Expect \'OCR\' '
                                  f'or \'STRUCTURE\'.')

    model_urls = MODEL_URLS[type_]
    if version not in model_urls:
        version = default_model_version
    if model_type not in model_urls[version]:
        if model_type in model_urls[default_model_version]:
            version = default_model_version
        else:
            raise Exception(f'{model_type} models is not support, we only support '
                            f'{model_urls[default_model_version].keys()}')

    if lang not in model_urls[version][model_type]:
        if lang in model_urls[default_model_version][model_type]:
            version = default_model_version
        else:
            raise Exception(
                f'Language {lang} is not support, we only support '
                f'{model_urls[default_model_version][model_type].keys()} for '
                f'{model_type} models'
            )
    return model_urls[version][model_type][lang]


def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def check_img(img_file_or_byte):
    if isinstance(img_file_or_byte, bytes):
        img = img_decode(img_file_or_byte)
    elif isinstance(img_file_or_byte, str):
        image_fn = img_file_or_byte
        img, flag_gif, flag_pdf = check_and_read(image_fn)
        if not flag_gif and not flag_pdf:
            with open(image_fn, 'rb') as g:
                img_str = g.read()
                img = img_decode(img_str)
            if img is None:
                try:
                    buf = BytesIO()
                    image = BytesIO(img_str)
                    im = Image.open(image)
                    rgb = im.convert('RGB')
                    rgb.save(buf, 'jpeg')
                    buf.seek(0)
                    image_bytes = buf.read()
                    data_base64 = str(base64.b64encode(image_bytes),
                                      encoding="utf-8")
                    image_decode = base64.b64decode(data_base64)
                    img_array = np.frombuffer(image_decode, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    logger.error(f"error in loading image: {image_fn}")
                    return None
        if img is None:
            logger.error(f"error in loading image: {image_fn}")
            return None
    elif isinstance(img_file_or_byte, np.ndarray) and len(img_file_or_byte.shape) == 2:
        img = cv2.cvtColor(img_file_or_byte, cv2.COLOR_GRAY2BGR)
    else:
        img = None
        logger.error("error in loading image.")
    return img


def url_to_filename(url):
    fn_ext = url.split('/')[-1]
    fn, _ = os.path.splitext(fn_ext)
    return fn


class TarDownloader:
    def __init__(self, url, output_dir):
        # Remove existed (potentially broken) model directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Calc zipped file path
        tar_fn = url.split('/')[-1]
        tar_fp = os.path.join(output_dir, tar_fn)

        # Calc file size
        response = requests.get(url, stream=True)
        response.raise_for_status()
        file_size = response.headers.get('Content-Length')
        try:
            file_size = int(file_size)
        except ValueError:
            raise Exception(f"Unknown file size of {tar_fn}")

        self.file_size = file_size
        self.response = response
        self.model_dir = output_dir
        self.tar_fp = tar_fp

    def download(self, progress_bar, current_size, total_size):
        with open(self.tar_fp, 'wb') as f:
            for chunk in self.response.iter_content(chunk_size=8192):
                f.write(chunk)
                current_size += len(chunk)
                progress_bar.emit(round(current_size / total_size * 100))
        return current_size

    def extract_and_delete(self):
        with tarfile.open(self.tar_fp, 'r:*') as tar:
            members = tar.getmembers()
            top_level_dirs = set()
            for m in members:
                if not m.name or m.name.startswith('/') or m.name.startswith('./'):
                    continue
                parts = m.name.split('/', 1)
                top_level_dirs.add(parts[0])
                if len(top_level_dirs) > 1:
                    break
            strip = len(top_level_dirs) == 1
            for m in members:
                if strip:
                    parts = m.name.split('/', 1)
                    if len(parts) == 2:
                        m.name = parts[1]
                    else:
                        continue  # Skip top-level directory itself
                tar.extract(m, path=self.model_dir)
        os.remove(self.tar_fp)


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, progress_bar, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args()
        params.__dict__.update(**kwargs)
        assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, (
            f"ocr_version must in {SUPPORT_OCR_MODEL_VERSION}, but get "
            f"{params.ocr_version}")

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls

        rec_lang = params.lang
        if rec_lang not in MODEL_URLS['OCR']['PP-OCRv4']['rec'].keys():
            rec_lang = 'ch'

        if rec_lang == 'ch':
            det_lang = 'ch'
        elif rec_lang in ('en', 'latin'):
            det_lang = 'en'
        else:
            det_lang = 'ml'

        # remove model URL
        det_url = MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['det'][det_lang]['url']
        rec_url = MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['rec'][rec_lang]['url']
        cls_url = MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['cls']['ch']['url']

        # local model path
        params.det_model_dir = os.path.join(
            BASE_DIR, 'whl', 'det', det_lang, url_to_filename(det_url))
        params.rec_model_dir = os.path.join(
            BASE_DIR, 'whl', 'rec', rec_lang, url_to_filename(rec_url))
        params.cls_model_dir = os.path.join(
            BASE_DIR, 'whl', 'cls', url_to_filename(cls_url))

        # image resolution
        params.rec_image_shape = "3, 48, 320"

        # download models
        assert params.use_onnx is False, "ONNX inference models are not supported."
        download_det = not all([
            os.path.exists(os.path.join(params.det_model_dir, "inference.pdiparams")),
            os.path.exists(os.path.join(params.det_model_dir, "inference.pdiparams.info")),
            os.path.exists(os.path.join(params.det_model_dir, "inference.pdmodel")),
        ])
        download_rec = not all([
            os.path.exists(os.path.join(params.rec_model_dir, "inference.pdiparams")),
            os.path.exists(os.path.join(params.rec_model_dir, "inference.pdiparams.info")),
            os.path.exists(os.path.join(params.rec_model_dir, "inference.pdmodel")),
        ])
        download_cls = not all([
            os.path.exists(os.path.join(params.cls_model_dir, "inference.pdiparams")),
            os.path.exists(os.path.join(params.cls_model_dir, "inference.pdiparams.info")),
            os.path.exists(os.path.join(params.cls_model_dir, "inference.pdmodel")),
        ])

        total_size = 0
        if download_det:
            downloader_det = TarDownloader(det_url, params.det_model_dir)
            total_size += downloader_det.file_size
        if download_rec:
            downloader_rec = TarDownloader(rec_url, params.rec_model_dir)
            total_size += downloader_rec.file_size
        if download_cls:
            downloader_cls = TarDownloader(cls_url, params.cls_model_dir)
            total_size += downloader_cls.file_size

        current_size = 0
        if download_det:
            current_size = downloader_det.download(progress_bar, current_size, total_size)
            downloader_det.extract_and_delete()
        if download_rec:
            current_size = downloader_rec.download(progress_bar, current_size, total_size)
            downloader_rec.extract_and_delete()
        if download_cls:
            _ = downloader_cls.download(progress_bar, current_size, total_size)
            downloader_cls.extract_and_delete()

        if params.rec_char_dict_path is None:
            char_dict_path = MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['rec'][rec_lang]['dict_path']
            params.rec_char_dict_path = os.path.join("paddleocr", char_dict_path)
        logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)

    def ocr(self, img, det=True, rec=True, cls=True, bin_=False, inv=False,
            alpha_color=(255, 255, 255)):
        """
        OCR with PaddleOCR
        args：
            img: img for OCR, support ndarray, img_path and list or ndarray
            det: use text detection or not. If False, only rec will be exec. Default is True
            rec: use text recognition or not. If False, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If True, the text with
            rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees,
            use cls=False to get better performance. Text with rotation of 90 or 270 degrees
            can be recognized even if cls=False.
            bin: binarize image to black and white. Default is False.
            inv: invert image colors. Default is False.
            alpha_color: set RGB color Tuple for transparent parts replacement. Default is
            pure white.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, it will not be used during the forward process'
            )
        img = check_img(img)

        def preprocess_image(_image):
            _image = alpha_to_color(_image, alpha_color)
            if inv:
                _image = cv2.bitwise_not(_image)
            if bin_:
                _image = binarize_img(_image)
            return _image

        if det and rec:
            img = preprocess_image(img)
            dt_boxes, rec_res, _ = self.detect_and_recognize(img, cls)
            if not dt_boxes and not rec_res:
                return
            tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            return tmp_res
        elif det and (not rec):
            img = preprocess_image(img)
            dt_boxes, elapse = self.text_detector(img)
            if not dt_boxes:
                return
            tmp_res = [box.tolist() for box in dt_boxes]
            return tmp_res
        elif (not det) and rec:
            img = [preprocess_image(img)]
            if self.use_angle_cls and cls:
                img, cls_res_tmp, elapse = self.text_classifier(img)
            rec_res, elapse = self.text_recognizer(img)
            return rec_res
        else: # not det, not rec
            cls_res = []
            img = [preprocess_image(img)]
            if self.use_angle_cls and cls:
                img, cls_res_tmp, elapse = self.text_classifier(img)
                cls_res.append(cls_res_tmp)
            return cls_res
