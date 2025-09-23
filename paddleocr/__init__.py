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
import logging
import os
import shutil
import tarfile
from io import BytesIO

import cv2
import numpy as np
import requests
import yaml
from PIL import Image

from paddleocr.default_arguments import parse_args
from paddleocr.ppocr.utils.logging import get_logger
from paddleocr.ppocr.utils.utility import (check_and_read, get_image_file_list,
                                           alpha_to_color, binarize_img)
from paddleocr.tools.infer import predict_system
from paddleocr.tools.infer.utility import draw_ocr, str2bool, get_model_files

logger = get_logger()
base_dir = os.path.abspath("./inference_models")
os.makedirs(base_dir, exist_ok=True)
with open("./paddleocr/model_urls.json", "r") as f:
    MODEL_URLS = yaml.safe_load(f)


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

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls

        rec_lang_all = MODEL_URLS['rec'].keys()
        assert params.lang in rec_lang_all, \
            (f"Character set of text recognition {params.lang} not supported. Supported "
             f"items: {rec_lang_all}")
        dict_all = MODEL_URLS['dict'].keys()
        assert params.lang in dict_all, \
            (f"Character set of dictionary {params.lang} not supported. Supported "
             f"items: {dict_all}")
        det_lang_all = MODEL_URLS['det'].keys()
        match params.lang:
            case "ch":
                det_lang = "ch"
            case "en" | "latin":
                det_lang = "en"
            case "universal":
                det_lang = "universal"
            case _:
                det_lang = "ml"
        assert det_lang in det_lang_all, \
            (f"Character set of text detection {det_lang} not supported. Supported items: "
             f"{det_lang_all}")

        # model URL
        det_url = MODEL_URLS['det'][det_lang]
        rec_url = MODEL_URLS['rec'][params.lang]
        cls_url = MODEL_URLS['cls']

        # local model path
        params.det_model_dir = os.path.join(
            base_dir, 'whl', 'det', det_lang, url_to_filename(det_url))
        params.rec_model_dir = os.path.join(
            base_dir, 'whl', 'rec', params.lang, url_to_filename(rec_url))
        params.cls_model_dir = os.path.join(
            base_dir, 'whl', 'cls', url_to_filename(cls_url))

        # image resolution
        params.rec_image_shape = "3, 48, 320"

        # download models
        assert params.use_onnx is False, "ONNX inference models are not supported."
        try:
            get_model_files(params.det_model_dir, "inference")
        except AssertionError:
            download_det = True
        else:
            download_det = False
        try:
            get_model_files(params.rec_model_dir, "inference")
        except AssertionError:
            download_rec = True
        else:
            download_rec = False
        try:
            get_model_files(params.cls_model_dir, "inference")
        except AssertionError:
            download_cls = True
        else:
            download_cls = False

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

        char_dict_path = MODEL_URLS['dict'][params.lang]
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
