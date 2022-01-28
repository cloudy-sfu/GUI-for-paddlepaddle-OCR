import os
import cv2
import numpy as np
from PyQt5.QtCore import *
from text_detection import detect_text
from text_recognition import recognize_text


class FolderOCR(QThread):
    progress = pyqtSignal(tuple)
    error_message = pyqtSignal(str)
    done = pyqtSignal(bool)

    def __init__(self, input_folder, output_folder):
        super(FolderOCR, self).__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder

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
            filenames += [(parent, x) for x in files if os.path.splitext(x)[1].lower() in ['.jpg', '.png']]
        n_files = len(filenames)
        if n_files == 0:
            self.error_message.emit('This folder doesn\'t contains any picture.')
            self.done.emit(True)
            return
        # OCR
        for i, (parent, name) in zip(range(n_files), filenames):
            filepath = os.path.join(parent, name)
            output_image_name, output_image_ext = os.path.splitext(name)
            output_path_text = os.path.join(self.output_folder, 'text', output_image_name + '.txt')
            output_path_image = os.path.join(self.output_folder, 'markup', name)
            try:
                image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)[:, :, :3]
                boxes = detect_text(image)
                texts = recognize_text(image, boxes)
            except Exception as e:
                self.error_message.emit(f'{filepath} - {e}')
                self.progress.emit((i + 1, n_files))
                continue
            if not texts:
                self.error_message.emit(f'{filepath} doesn\'t contain texts.')
            with open(output_path_text, 'w') as f:
                for block in texts:
                    f.write(block['text'] + '\n')
                    image = np.ascontiguousarray(image, dtype=np.uint8)
                    cv2.rectangle(image, block['text_box_position'][0], block['text_box_position'][2], (0, 0, 255))
                    cv2.putText(image, str(round(block['confidence'], 2)), block['text_box_position'][0],
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            cv2.imencode(output_image_ext.lower(), image)[1].tofile(output_path_image)
            del image, texts, boxes
            self.progress.emit((i + 1, n_files))
        self.done.emit(True)
