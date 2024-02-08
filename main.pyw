import shutil
import sys
from PyQt5.Qt import *
from PyQt5.QtWidgets import *
from ocr import *
from PIL import Image
from csv import reader as csv_reader


with open("languages.csv", mode='r', encoding='utf-8') as f:
    languages = {
        row[0]: row[1]
        for row in csv_reader(f)
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
    q_image = q_image.convertToFormat(QImage.Format_RGBA8888)
    width = q_image.width()
    height = q_image.height()
    if q_image.bits() is None:
        return
    buffer = q_image.bits().asarray(q_image.byteCount())
    pillow_image = Image.frombytes("RGBA", (width, height),
                                   buffer, "raw", "BGRA")
    return pillow_image


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__(flags=Qt.Window)
        dpi = self.screen().logicalDotsPerInch()
        font_size = max(6, round(22 - dpi / 12))
        self.setStyleSheet(
            f'font-family: "Microsoft YaHei", Calibri, Ubuntu; font-size: {font_size}pt;')
        self.resize(1280, 720)
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

        main_part = QWidget(self)
        main_layout = QFormLayout(main_part)
        main_layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
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
        open_project_root = QAction('&Open backend files location', self)
        open_project_root.triggered.connect(self.open_inference_models)
        clear_models = QAction('&Clear inference models', self)
        clear_models.triggered.connect(self.clear_models)
        switch_language = QAction('&Switch language ...', self)
        switch_language.triggered.connect(self.switch_language)
        switch_language.setShortcut('Ctrl+L')
        force_idle = QAction('&Interrupt', self)
        force_idle.triggered.connect(self.interrupt_operator)
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
        file.addActions([open_project_root, switch_language, force_idle, clear_models,
                         close])
        recognize = QMenu('&Recognize', self)
        recognize.addActions([reco_folder, reco_file, reco_clipboard])
        # Menu bar
        menubar = QMenuBar()
        menubar.addMenu(file)
        menubar.addMenu(recognize)
        self.setMenuBar(menubar)

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
        fp = QFileDialog.getExistingDirectory(self, caption='Images to recognize',
                                              options=QFileDialog.ShowDirsOnly)
        if not (fp and os.path.isdir(fp)):
            self.message.append('The source to recognize does not exist.')
            self.delayed_thread_finished()
            return
        dist = QFileDialog.getExistingDirectory(self, caption='Export to',
                                                options=QFileDialog.ShowDirsOnly)
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
        fp, _ = QFileDialog.getOpenFileName(self, filter='Images (*.png *.jpeg *.jpg)')
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

    def open_inference_models():
        target = os.path.abspath("inference_models")
        os.makedirs(target, exist_ok=True)
        os.startfile(target)

    def center(self):
        fg = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        fg.moveCenter(cp)
        self.move(fg.topLeft())

    def create_language_box(self):
        self.language_combo = QComboBox()
        self.language_combo.addItems(list(languages.keys()))
        self.language_box = QDialog()
        dpi = self.screen().logicalDotsPerInch() / 96
        font_size = 14 if dpi <= 1 else (
            12 if 1 < dpi <= 1.25 else (10 if 1.25 < dpi <= 1.5 else 8))
        self.language_box.setStyleSheet(
            f'font-family: "Microsoft YaHei", Calibri, Ubuntu; font-size: {font_size}pt;')
        self.language_box.setMinimumWidth(480)
        self.language_box.setWindowTitle('Select a language')
        layout = QVBoxLayout()
        hint = QLabel('Please select a supported language.')
        hint.setWordWrap(True)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.language_box.accept)
        button_box.rejected.connect(self.language_box.reject)
        layout.addWidget(hint)
        layout.addWidget(self.language_combo)
        layout.addWidget(button_box)
        self.language_box.setLayout(layout)

    def set_ocr_engine(self, ocr_engine):
        self.ocr_engine = ocr_engine

    @status_check_decorator(action_name='Initialize OCR engine')
    def switch_language(self):
        action = self.language_box.exec_()
        if action != QDialog.Accepted:
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
        self.operator = InitializeModel(lang_abbr)
        self.operator.start()
        self.operator.ocr_engine.connect(self.set_ocr_engine)
        self.operator.gui_message.connect(self.message.append)
        self.operator.done.connect(self.delayed_thread_finished)

    @status_check_decorator(action_name="Clear models")
    def clear_models(self):
        shutil.rmtree('inference_models/whl/det', ignore_errors=True)
        shutil.rmtree('inference_models/whl/rec', ignore_errors=True)
        self.lang = None
        self.delayed_thread_finished()

    # Deceive IDE grammar warning; must be written end of the class.
    status_check_decorator = staticmethod(status_check_decorator)
    open_inference_models = staticmethod(open_inference_models)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myw = MyWindow()
    myw.show()
    sys.exit(app.exec_())
