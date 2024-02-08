import os
import cv2
import numpy as np
from PyQt5.QtCore import *
os.environ['PATH'] += ';' + os.path.abspath('mklml/lib')
from paddleocr import PaddleOCR
from paddleocr.ppocr.utils.utility import alpha_to_color


class FolderOCR(QThread):
    progress = pyqtSignal(int, name='progress')
    gui_message = pyqtSignal(str, name='error_message')
    done = pyqtSignal(bool, name='done')

    def __init__(self, input_folder, output_folder, ocr_engine):
        super(FolderOCR, self).__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.ocr_engine = ocr_engine

    def run(self) -> None:
        # build folder trees
        markup_folder = os.path.join(self.output_folder, 'markup')
        if not os.path.exists(markup_folder):
            os.mkdir(markup_folder)
        text_folder = os.path.join(self.output_folder, 'text')
        if not os.path.exists(text_folder):
            os.mkdir(text_folder)
        # get files' path
        filenames = []
        for parent, _, files in os.walk(self.input_folder):
            filenames += [(parent, x) for x in files if
                          os.path.splitext(x)[1].lower() in ['.jpg', '.png']]
        n_files = len(filenames)
        if n_files == 0:
            self.gui_message.emit('This folder doesn\'t contains any picture.')
            self.done.emit(True)
            return
        # batch logic
        for i, (parent, name) in zip(range(n_files), filenames):
            filepath = os.path.join(parent, name)
            output_image_name, output_image_ext = os.path.splitext(name)
            output_path_text = os.path.join(self.output_folder, 'text',
                                            output_image_name + '.txt')
            output_path_image = os.path.join(self.output_folder, 'markup', name)

            # OCR
            with open(filepath, 'rb') as g:
                img = g.read()
            try:
                img = np.frombuffer(img, dtype=np.uint8)  # TypeError if None
                img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)  # cv2.error if None
                img = alpha_to_color(img)  # AttributeError if None
                img = cv2.bitwise_not(img)
            except (cv2.error, TypeError, AttributeError):
                self.gui_message.emit(f"{filepath} - Image file is broken.")
                self.progress.emit((i + 1, n_files))
                del img
                continue
            try:
                boxes, texts = self.ocr_engine.detect_and_recognize(img)
            except Exception as e:
                self.gui_message.emit(f"{filepath} - {e}")
                self.progress.emit((i + 1, n_files))
                del img
                continue
            if not texts:
                self.gui_message.emit(f'{filepath} doesn\'t contain texts.')
            with open(output_path_text, 'w') as f:
                for box, text in zip(boxes, texts):
                    f.write(text[0] + '\n')
                    img = np.ascontiguousarray(img, dtype=np.uint8)
                    box = box.astype(int).tolist()
                    cv2.rectangle(img, box[0], box[2], (0, 0, 255))
                    cv2.putText(img, str(round(text[1], 2)), box[0],
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            cv2.imencode(output_image_ext.lower(), img)[1].tofile(output_path_image)
            del img, texts, boxes
            progress_percentage = int((i + 1) / n_files * 100)
            self.progress.emit(progress_percentage)
        os.startfile(self.output_folder)
        self.done.emit(True)


class FileOCR(QThread):
    progress = pyqtSignal(int, name='progress')
    gui_message = pyqtSignal(str, name='gui_message')
    done = pyqtSignal(bool, name='done')

    def __init__(self, input_file, ocr_engine):
        super(FileOCR, self).__init__()
        self.input_file = input_file
        self.ocr_engine = ocr_engine

    def run(self) -> None:
        with open(self.input_file, 'rb') as g:
            img = g.read()
        try:
            img = np.frombuffer(img, dtype=np.uint8)  # TypeError if None
            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)  # cv2.error if None
            img = alpha_to_color(img)  # AttributeError if None
            img = cv2.bitwise_not(img)
        except (cv2.error, TypeError, AttributeError):
            self.gui_message.emit(f"{self.input_file} - Image file is broken.")
            self.done.emit(True)
            del img
            return
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            boxes, texts = self.ocr_engine.detect_and_recognize(img)
        except Exception as e:
            self.gui_message.emit(f"{self.input_file} - {e}")
            self.done.emit(True)
            del img
            return
        for block in texts:
            self.gui_message.emit(block[0])
        del img, boxes, texts
        self.done.emit(True)


class ClipboardOCR(QThread):
    progress = pyqtSignal(int, name='progress')
    gui_message = pyqtSignal(str, name='gui_message')
    done = pyqtSignal(bool, name='done')

    def __init__(self, pillow_image, ocr_engine):
        super(ClipboardOCR, self).__init__()
        self.pillow_image = pillow_image
        self.ocr_engine = ocr_engine

    def run(self) -> None:
        try:
            img = np.array(self.pillow_image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = alpha_to_color(img)  # AttributeError if None
            img = cv2.bitwise_not(img)
        except (cv2.error, TypeError, AttributeError):
            self.gui_message.emit("Image file in clipboard is broken.")
            del img
            self.done.emit(True)
            return
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            boxes, texts = self.ocr_engine.detect_and_recognize(img)
        except Exception as e:
            self.gui_message.emit(e.__str__())
            del img
            self.done.emit(True)
            return
        for block in texts:
            self.gui_message.emit(block[0])
        del img, boxes, texts
        self.done.emit(True)


class InitializeModel(QThread):
    ocr_engine = pyqtSignal(PaddleOCR, name='ocr_engine')
    gui_message = pyqtSignal(str, name='gui_message')
    done = pyqtSignal(bool, name='done')

    def __init__(self, lang_abbr):
        super(InitializeModel, self).__init__()
        self.lang_abbr = lang_abbr

    def run(self) -> None:
        try:
            engine = PaddleOCR(
                use_angle_cls=True, lang=self.lang_abbr, show_log=False,
                det_algorithm='DB++', use_gpu=False, enable_mkldnn=False,
            )
            self.ocr_engine.emit(engine)
            self.gui_message.emit("The program has finished initializing the inference "
                                  "model.")
        except Exception as e:
            self.gui_message.emit(e.__str__())
        self.done.emit(True)
