import os
import cv2
import numpy as np
from PyQt5.QtCore import *
from text_detection import detect_text
from text_recognition import recognize_text


class FolderOCR(QThread):
    progress = pyqtSignal(int, name='progress')
    gui_message = pyqtSignal(str, name='error_message')
    done = pyqtSignal(bool, name='done')

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
            self.gui_message.emit('This folder doesn\'t contains any picture.')
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
                self.gui_message.emit(f'{filepath} - {e}')
                self.progress.emit((i + 1, n_files))
                continue
            if not texts:
                self.gui_message.emit(f'{filepath} doesn\'t contain texts.')
            with open(output_path_text, 'w') as f:
                for block in texts:
                    f.write(block['text'] + '\n')
                    image = np.ascontiguousarray(image, dtype=np.uint8)
                    cv2.rectangle(image, block['text_box_position'][0], block['text_box_position'][2], (0, 0, 255))
                    cv2.putText(image, str(round(block['confidence'], 2)), block['text_box_position'][0],
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            cv2.imencode(output_image_ext.lower(), image)[1].tofile(output_path_image)
            del image, texts, boxes
            progress_percentage = int((i + 1) / n_files * 100)
            self.progress.emit(progress_percentage)
        self.gui_message.emit('Click "File | Open target folder" in menu bar to view the exported images.')
        self.done.emit(True)


class FileOCR(QThread):
    progress = pyqtSignal(int, name='progress')
    gui_message = pyqtSignal(str, name='gui_message')
    done = pyqtSignal(bool, name='done')

    def __init__(self, input_file, output_folder):
        super(FileOCR, self).__init__()
        self.input_file = input_file
        self.output_folder = output_folder

    def run(self) -> None:
        try:
            image = cv2.imdecode(np.fromfile(self.input_file, dtype=np.uint8), -1)[:, :, :3]
            boxes = detect_text(image)
            texts = recognize_text(image, boxes)
        except Exception as e:
            self.gui_message.emit(f'{self.input_file} - {e}')
            self.done.emit(True)
            return
        for block in texts:
            self.gui_message.emit(block['text'])
        del image, boxes, texts
        self.progress.emit(100)
        self.done.emit(True)
