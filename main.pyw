import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ocr import FolderOCR, FileOCR


class MyWindow(QMainWindow):

    def __init__(self):
        super(MyWindow, self).__init__(flags=Qt.Window)
        dpi = self.screen().logicalDotsPerInch()
        font_size = max(6, round(22 - dpi / 12))
        os.makedirs('raw/clipboard/', exist_ok=True)
        self.setStyleSheet(f'font-family: "Microsoft YaHei", Calibri, Ubuntu; font-size: '
                           f'{font_size}pt;')
        self.resize(800, 480)
        self.setWindowTitle('GUI for paddlepaddle OCR')
        self.center()

        self.operator1 = None
        self.operator2 = None
        self.busy = False
        self.project_root = os.path.abspath('raw/')

        self.create_menu_bar()

        self.project_root_displayed = QLabel(self)
        self.project_root_displayed.setText(self.project_root)
        self.input_anytype_displayed = QLabel(self)
        self.pbar = QProgressBar(self)
        self.message = QTextEdit(self)
        self.message.setText("paddlepaddleocr-v4-pyqt-5 Note: \n"
                             "When you first infer with a new language, the program takes "
                             "several minutes to download the inference model (depending "
                             "to your Internet bandwidth). The progress bar is not made "
                             "at the moment. Thank you for your patience. To check whether "
                             "the downloading is stuck, please monitor the size of "
                             "\"inference_models\" directory in the installation folder. "
                             "It is expected to expand when downloading. If it stops "
                             "expanding for several minutes, please restart this program.")

        main_part = QWidget(self)
        main_layout = QFormLayout(main_part)
        main_layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        main_layout.addRow('Source:', self.input_anytype_displayed)
        main_layout.addRow('Target:', self.project_root_displayed)
        main_layout.addRow('Progress:', self.pbar)
        main_layout.addRow('Message:', self.message)
        self.setCentralWidget(main_part)

        self.status = QStatusBar()
        self.status.showMessage('Ready.', 0)
        self.setStatusBar(self.status)

    def create_menu_bar(self):
        menubar = QMenuBar()

        # File menu
        open_project_root_action = QAction('&Open target folder ...', self)
        open_project_root_action.triggered.connect(self.open_project_root)

        close_program = QAction('&Exit', self)
        close_program.triggered.connect(self.close)
        close_program.setShortcut('Alt+F4')

        file_menu = QMenu('&File', self)
        file_menu.addActions([open_project_root_action, close_program])
        menubar.addMenu(file_menu)

        # Recognize menu
        open_folder = QAction('From &folder ...', self)
        open_folder.triggered.connect(self.ocr_batch)
        open_folder.setShortcut('Ctrl+T')

        open_file = QAction('From &single image ...', self)
        open_file.triggered.connect(self.ocr_single)

        paste_from_clipboard = QAction('From &clipboard', self)
        paste_from_clipboard.triggered.connect(self.ocr_clipboard)
        paste_from_clipboard.setShortcut('Ctrl+Shift+V')

        run_menu = QMenu('&Recognize', self)
        run_menu.addActions([open_folder, open_file, paste_from_clipboard])
        menubar.addMenu(run_menu)

        self.setMenuBar(menubar)

    def status_check_decorator(action_name, *args, **kwargs):
        def status_check_decorator_1(pyfunc):
            def status_check(self):
                if not self.busy:
                    self.busy = True
                    self.status.showMessage(f'{action_name} ...', 0)
                    self.message.clear()
                    pyfunc(self, *args, **kwargs)
                    self.busy = False
                    self.status.showMessage('Ready.', 0)
                else:
                    self.status.showMessage('The program is busy ...', 0)

            return status_check

        return status_check_decorator_1

    def delayed_thread_check_decorator(action_name, *args, **kwargs):
        def delayed_thread_check_decorator_1(pyfunc):
            def delayed_thread_check(self):
                if not self.busy:
                    self.busy = True
                    self.status.showMessage(f'{action_name} ...', 0)
                    self.message.clear()
                    pyfunc(self, *args, **kwargs)
                else:
                    self.status.showMessage('The program is busy ...', 0)

            return delayed_thread_check

        return delayed_thread_check_decorator_1

    @delayed_thread_check_decorator(action_name='Recognize folder')
    def ocr_batch(self):
        fp = QFileDialog.getExistingDirectory(self, caption='Images to recognize',
                                              options=QFileDialog.ShowDirsOnly)
        if not (fp and os.path.isdir(fp)):
            self.message.append('The source to recognize does not exist.')
            self.delayed_thread_finished()
            return
        dist = QFileDialog.getExistingDirectory(self, caption='Export to',
                                                options=QFileDialog.ShowDirsOnly)
        if not (dist and os.path.exists(dist)):
            self.message.append('The target to export does not exist.')
            self.delayed_thread_finished()
            return
        self.project_root = dist
        self.input_anytype_displayed.setText(fp)
        self.operator1 = FolderOCR(fp, dist)
        self.operator1.start()
        self.operator1.progress.connect(lambda x: self.pbar.setValue(x))
        self.operator1.gui_message.connect(lambda x: self.message.append(x))
        self.operator1.done.connect(lambda x: self.delayed_thread_finished())

    @delayed_thread_check_decorator(action_name='Recognize from clipboard')
    def ocr_clipboard(self):
        clipboard = QApplication.clipboard()
        picture = clipboard.pixmap()
        fp = os.path.join('raw/clipboard', f'hash_{hash(picture)}.png')
        if not picture.save(fp):
            self.message.append('No picture is detected.')
            self.delayed_thread_finished()
            return
        self.input_anytype_displayed.setText(fp)
        self.operator2 = FileOCR(fp, 'raw/')
        self.operator2.start()
        self.operator2.progress.connect(lambda x: self.pbar.setValue(x))
        self.operator2.gui_message.connect(lambda x: self.message.append(x))
        self.operator2.done.connect(lambda x: self.delayed_thread_finished())

    @delayed_thread_check_decorator(action_name='OCR for single image')
    def ocr_single(self):
        fp, _ = QFileDialog.getOpenFileName(self, filter='Images (*.png *.jpeg *.jpg)')
        if not (fp and os.path.isfile(fp)):
            self.message.append('The image to recognize does not exist.')
            self.delayed_thread_finished()
            return
        self.input_anytype_displayed.setText(fp)

        self.operator2 = FileOCR(fp, 'raw/')
        self.operator2.start()
        self.operator2.progress.connect(lambda x: self.pbar.setValue(x))
        self.operator2.gui_message.connect(lambda x: self.message.append(x))
        self.operator2.done.connect(lambda x: self.delayed_thread_finished())

    @status_check_decorator(action_name='Open target folder')
    def open_project_root(self):
        if not os.path.exists(self.project_root):
            self.message.append('The target folder does not exist.')
        q_url = QUrl()
        QDesktopServices.openUrl(q_url.fromLocalFile(self.project_root))

    def delayed_thread_finished(self):
        self.busy = False
        self.status.showMessage('Ready.', 0)

    def center(self):
        fg = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        fg.moveCenter(cp)
        self.move(fg.topLeft())

    # Deceive IDE grammar warning; must be written end of the class.
    status_check_decorator = staticmethod(status_check_decorator)
    delayed_thread_check_decorator = staticmethod(delayed_thread_check_decorator)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myw = MyWindow()
    myw.show()
    sys.exit(app.exec_())
