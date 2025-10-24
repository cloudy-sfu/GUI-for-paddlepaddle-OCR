import shutil
import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QAction, QImage
from PyQt6.QtCore import Qt
from ocr import *
from PIL import Image

languages = {
    'Chinese & Japanese & Latin (basic)': 'universal',
    'Korean & Latin (basic)': 'korean',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kabardian': 'ka',
    'Latin (extended)': 'latin',
    'Arabic': 'arabic',
    'Cyrillic & Latin (basic)': 'cyrillic',
    'Devanagari': 'devanagari',
    'Thai & Latin (basic)': 'th',
    'Greek & Latin (basic)': 'el',
    'Latin (basic)': 'en',
}
mirrors = {
    "Baidu cloud (CN)": None,
    "GitHub": lambda name:
    f"https://raw.githubusercontent.com/cloudy-sfu/GUI-for-paddlepaddle-OCR/refs/heads/"
    f"paddleocr-v5/models/{name}.tar"
}

not_initialized_message = ("OCR engine hasn't been initialized. The possible reasons "
                           "are as follows:\n"
                           "(1) If \"Language\" item is blank, then you haven't selected "
                           "a inference model. Please click \"File\" menu tab and then \""
                           "Switch language\" item. The program will initialize the "
                           "model.\n"
                           "(2) Last time when the program downloaded the model, the "
                           "process is interrupted. Please wait for 1 minute, and "
                           "paddlepaddle will retry downloading the model.\n"
                           "The downloading speed is dependent to your Internet "
                           "bandwidth. If you have IT background, please monitor the "
                           "size of \"inference_model\" folder. It is expected to expand "
                           "gradually during downloading. If you're unsure about the "
                           "technique things, please report an issue at "
                           "https://github.com/cloudy-sfu/GUI-for-paddlepaddle-OCR/issues "
                           "Our community may have a specific explanation and solution.\n"
                           "(3) If you have selected a language and waited for "
                           "reasonable time (1 minute), but the model is still not "
                           "initialized, it means paddlepaddle is trapped by broken "
                           "models. Please make sure the \"Language\" item shows the "
                           "language that you intend to recognize, click \"File\" "
                           "menu tab, then \"Clear models\" tab. \n")


def pixmap_to_pillow_image(pixmap):
    q_image = pixmap.toImage()
    q_image = q_image.convertToFormat(QImage.Format.Format_RGBA8888)
    width = q_image.width()
    height = q_image.height()
    if q_image.bits() is None:
        return None
    buffer = q_image.bits().asarray(q_image.sizeInBytes())
    pillow_image = Image.frombytes("RGBA", (width, height),
                                   buffer, "raw", "BGRA")
    return pillow_image


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__(flags=Qt.WindowType.Window)
        self.resize(960, 540)
        self.setWindowTitle('GUI for paddlepaddle OCR')
        self.center()

        self.busy = False
        self.operator = None
        self.ocr_engine = None
        self.lang = None

        self.create_menu_bar()
        self.create_language_box()
        self.source_displayed = QLabel(self)
        self.lang_displayed = QLabel(self)
        self.pbar = QProgressBar(self)
        self.message = QTextEdit(self)

        main_part = QWidget()
        main_layout = QFormLayout(main_part)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        main_layout.addRow('Source:', self.source_displayed)
        main_layout.addRow('Language:', self.lang_displayed)
        main_layout.addRow('Progress:', self.pbar)
        main_layout.addRow('Message:', self.message)
        self.setCentralWidget(main_part)

        self.status = QStatusBar()
        self.status.showMessage('Ready.', 0)
        self.setStatusBar(self.status)

    def create_menu_bar(self):
        # File menu
        open_models_fp = QAction('&Open inference models folder', self)
        open_models_fp.triggered.connect(self.open_inference_models)
        clear_models = QAction('&Clear inference models', self)
        clear_models.triggered.connect(self.clear_models)
        switch_lang = QAction('&Switch language ...', self)
        switch_lang.triggered.connect(self.switch_language)
        switch_lang.setShortcut('Ctrl+L')
        # interrupt = QAction('&Interrupt', self)
        # interrupt.triggered.connect(self.interrupt_operator)
        close = QAction('&Exit', self)
        close.triggered.connect(self.close)
        close.setShortcut('Ctrl+W')
        # Recognize menu
        reco_folder = QAction('From &folder ...', self)
        reco_folder.triggered.connect(self.ocr_batch)
        reco_file = QAction('From &single image ...', self)
        reco_file.triggered.connect(self.ocr_single)
        reco_clipboard = QAction('From &clipboard', self)
        reco_clipboard.triggered.connect(self.ocr_clipboard)
        reco_clipboard.setShortcut('Ctrl+Shift+V')
        # First-level buttons
        file = QMenu('&File', self)
        # file.addActions([open_models_fp, clear_models, switch_lang, interrupt, close])
        file.addActions([open_models_fp, clear_models, switch_lang, close])
        recognize = QMenu('&Recognize', self)
        recognize.addActions([reco_folder, reco_file, reco_clipboard])
        # Menu bar
        menubar = QMenuBar()
        menubar.addMenu(file)
        menubar.addMenu(recognize)
        self.setMenuBar(menubar)

    @staticmethod
    def status_check_decorator(action_name, *args, **kwargs):
        def status_check_decorator_1(pyfunc):
            def status_check(self):
                if not self.busy:
                    self.busy = True
                    self.status.showMessage(f'{action_name} ...', 0)
                    self.pbar.setValue(0)
                    self.message.clear()
                    pyfunc(self, *args, **kwargs)
            return status_check

        return status_check_decorator_1

    def delayed_thread_finished(self):
        self.pbar.setValue(100)
        self.status.showMessage('Ready.', 0)
        self.busy = False

    def interrupt_operator(self):
        if self.operator is not None:
            self.operator.exit(0)
            self.status.showMessage('Ready.', 0)
            self.pbar.setValue(0)
            self.busy = False

    @status_check_decorator(action_name='Recognize folder')
    def ocr_batch(self):
        if self.ocr_engine is None:
            self.message.append(not_initialized_message)
            self.delayed_thread_finished()
            return
        fp = QFileDialog.getExistingDirectory(
            caption='Images to recognize', options=QFileDialog.Option.ShowDirsOnly)
        if not (fp and os.path.isdir(fp)):
            self.message.append('The source to recognize does not exist.')
            self.delayed_thread_finished()
            return
        dist = QFileDialog.getExistingDirectory(
            caption='Export to', options=QFileDialog.Option.ShowDirsOnly)
        os.makedirs(dist, exist_ok=True)
        self.source_displayed.setText(fp)
        self.operator = FolderOCR(fp, dist, self.ocr_engine)
        self.operator.start()
        self.operator.progress.connect(self.pbar.setValue)
        self.operator.gui_message.connect(self.message.append)
        self.operator.done.connect(self.delayed_thread_finished)

    @status_check_decorator(action_name='Recognize from clipboard')
    def ocr_clipboard(self):
        if self.ocr_engine is None:
            self.message.append(not_initialized_message)
            self.delayed_thread_finished()
            return
        clipboard = QApplication.clipboard()
        pixmap = clipboard.pixmap()
        image = pixmap_to_pillow_image(pixmap)
        if not image:
            self.message.append('There is no picture in the clipboard.')
            self.delayed_thread_finished()
            return
        self.source_displayed.setText('Clipboard')
        self.operator = ClipboardOCR(image, self.ocr_engine)
        self.operator.start()
        self.operator.progress.connect(self.pbar.setValue)
        self.operator.gui_message.connect(self.message.append)
        self.operator.done.connect(self.delayed_thread_finished)

    @status_check_decorator(action_name='OCR for single image')
    def ocr_single(self):
        if self.ocr_engine is None:
            self.message.append(not_initialized_message)
            self.delayed_thread_finished()
            return
        fp, _ = QFileDialog.getOpenFileName(filter='Images (*.png *.jpeg *.jpg)')
        if not (fp and os.path.isfile(fp)):
            self.message.append('The image to recognize does not exist.')
            self.delayed_thread_finished()
            return
        self.source_displayed.setText(fp)
        self.operator = FileOCR(fp, self.ocr_engine)
        self.operator.start()
        self.operator.progress.connect(self.pbar.setValue)
        self.operator.gui_message.connect(self.message.append)
        self.operator.done.connect(self.delayed_thread_finished)

    @staticmethod
    def open_inference_models():
        target = os.path.abspath("inference_models")
        os.makedirs(target, exist_ok=True)
        os.startfile(target)

    def center(self):
        fg = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        fg.moveCenter(cp)
        self.move(fg.topLeft())

    def create_language_box(self):
        # Dialog body
        self.language_box = QDialog()
        self.language_box.setMinimumWidth(360)
        self.language_box.setWindowTitle('Language')
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.language_box.accept)
        button_box.rejected.connect(self.language_box.reject)
        layout = QVBoxLayout()
        # Content
        lang_label = QLabel('Please select a supported character set.')
        lang_label.setWordWrap(True)
        layout.addWidget(lang_label)
        self.language_combo = QComboBox()
        self.language_combo.addItems(list(languages.keys()))
        layout.addWidget(self.language_combo)
        mirror_label = QLabel('Please select the download source.')
        mirror_label.setWordWrap(True)
        layout.addWidget(mirror_label)
        self.mirror_combo = QComboBox()
        self.mirror_combo.addItems(list(mirrors.keys()))
        layout.addWidget(self.mirror_combo)
        # Dialog body, post-content
        layout.addWidget(button_box)
        self.language_box.setLayout(layout)

    def set_ocr_engine(self, ocr_engine):
        self.ocr_engine = ocr_engine

    @status_check_decorator(action_name='Initialize OCR engine')
    def switch_language(self):
        action = self.language_box.exec()
        if action != QDialog.DialogCode.Accepted:
            self.delayed_thread_finished()
            return
        self.lang = self.language_combo.currentText()
        self.lang_displayed.setText(self.lang)
        if self.lang is None:
            self.delayed_thread_finished()
            return
        self.message.append("The program is downloading and initializing the "
                            "inference model.")
        lang_abbr = languages[self.lang]
        mirror_key = self.mirror_combo.currentText()
        mirror = mirrors[mirror_key]
        self.operator = InitializeModel(lang_abbr, mirror)
        self.operator.start()
        self.operator.ocr_engine.connect(self.set_ocr_engine)
        self.operator.gui_message.connect(self.message.append)
        self.operator.done.connect(self.delayed_thread_finished)
        self.operator.progress.connect(self.pbar.setValue)

    @status_check_decorator(action_name="Clear models")
    def clear_models(self):
        shutil.rmtree('inference_models/det', ignore_errors=True)
        shutil.rmtree('inference_models/rec', ignore_errors=True)
        shutil.rmtree('inference_models/cls', ignore_errors=True)
        self.lang = None
        self.delayed_thread_finished()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    # Windows, Linux
    zoom_pct = screen.logicalDotsPerInch() / 96
    font_size = round(12 * zoom_pct)
    app.setStyleSheet(
        f'QWidget {{'
        f'    font-family: "Microsoft YaHei", Calibri, Ubuntu; '
        f'    font-size: {font_size}pt;'
        f'}}'
    )

    myw = MyWindow()
    myw.show()
    sys.exit(app.exec())
