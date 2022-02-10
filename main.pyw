import json
import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ocr import FolderOCR, FileOCR


class MyWindow(QMainWindow):

    def __init__(self):
        super(MyWindow, self).__init__(flags=Qt.Window)
        dpi = self.screen().logicalDotsPerInch() / 96
        font_size = 14 if dpi <= 1 else (12 if 1 < dpi <= 1.25 else (10 if 1.25 < dpi <= 1.5 else 8))
        with open('inference_model/project.json', 'r') as f:
            self.config = json.load(f)

        self.setStyleSheet(f'font-family: "Microsoft YaHei", Calibri, Ubuntu; font-size: {font_size}pt;')
        self.resize(800, 480)
        self.setWindowTitle('Chinese and English OCR')
        self.center()

        self.operator1 = None
        self.operator2 = None
        self.busy = False

        self.create_menu_bar()

        self.project_root_displayed = QLabel(self)
        self.project_root_displayed.setText(self.config['project_root'])
        self.input_anytype_displayed = QLabel(self)
        self.pbar = QProgressBar(self)
        self.message = QTextEdit(self)

        main_part = QWidget(self)
        main_layout = QFormLayout(main_part)
        main_layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        main_layout.addRow('Project:', self.project_root_displayed)
        main_layout.addRow('Opened:', self.input_anytype_displayed)
        main_layout.addRow('Progress:', self.pbar)
        main_layout.addRow('Message:', self.message)
        self.setCentralWidget(main_part)

        self.status = QStatusBar()
        self.status.showMessage('Ready.', 0)
        self.setStatusBar(self.status)

    def create_menu_bar(self):
        menubar = QMenuBar()

        # File menu
        new_project = QAction('&New project ...', self)
        new_project.triggered.connect(self.define_project)
        new_project.setShortcut('Ctrl+T')

        open_project_root_action = QAction('&Open project folder ...', self)
        open_project_root_action.triggered.connect(self.open_project_root)

        close_program = QAction('&Exit', self)
        close_program.triggered.connect(self.save_and_close)
        close_program.setShortcut('Alt+F4')

        file_menu = QMenu('&File', self)
        file_menu.addActions([new_project, open_project_root_action, close_program])
        menubar.addMenu(file_menu)

        # Recognize menu
        open_folder = QAction('From &folder ...', self)
        open_folder.triggered.connect(self.ocr_batch)

        open_file = QAction('From &single image ...', self)
        open_file.triggered.connect(self.ocr_single)

        paste_from_clipboard = QAction('From &clipboard', self)
        paste_from_clipboard.triggered.connect(self.ocr_clipboard)
        paste_from_clipboard.setShortcut('Ctrl+Shift+V')

        run_menu = QMenu('&Recognize', self)
        run_menu.addActions([open_folder, open_file, paste_from_clipboard])
        menubar.addMenu(run_menu)

        # About menu
        about_the_author = QAction('&About the author', self)
        about_the_author.triggered.connect(self.print_author_info)

        about_menu = QMenu('&About', self)
        about_menu.addActions([about_the_author])
        menubar.addMenu(about_menu)

        self.setMenuBar(menubar)

    def save_and_close(self):
        with open('inference_model/project.json', 'w') as f:
            json.dump(self.config, f)
        self.close()

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

    @status_check_decorator(action_name='Define project')
    def define_project(self):
        self.config['project_root'] = QFileDialog.getExistingDirectory(self)
        self.project_root_displayed.setText(self.config['project_root'])

    @delayed_thread_check_decorator(action_name='Recognize folder')
    def ocr_batch(self):
        if not self.config['project_root']:
            self.message.append('Project is not defined.')
            self.delayed_thread_finished()
            return

        fp = QFileDialog.getExistingDirectory(self)
        if not (fp and os.path.exists(fp) and fp != self.config['project_root']):
            self.message.append('The folder to recognize does not exist.')
            self.delayed_thread_finished()
            return

        self.input_anytype_displayed.setText(fp)
        self.operator1 = FolderOCR(fp, self.config['project_root'])
        self.operator1.start()
        self.operator1.progress.connect(lambda x: self.pbar.setValue(x))
        self.operator1.error_message.connect(lambda x: self.message.append(x))
        self.operator1.done.connect(lambda x: self.delayed_thread_finished())

    @delayed_thread_check_decorator(action_name='Recognize from clipboard')
    def ocr_clipboard(self):
        if not self.config['project_root']:
            self.message.append('Project is not defined.')
            self.delayed_thread_finished()
            return

        clipboard_saved_path = os.path.join(self.config['project_root'], 'origin_clipboard')
        if not os.path.exists(clipboard_saved_path):
            os.mkdir(clipboard_saved_path)
        clipboard = QApplication.clipboard()
        picture = clipboard.pixmap()
        fp = os.path.join(clipboard_saved_path, f'hash_{hash(picture)}.png')
        if not picture.save(fp):
            self.message.append('No picture is detected.')
            self.delayed_thread_finished()
            return
        self.input_anytype_displayed.setText(fp)

        self.operator2 = FileOCR(fp, self.config['project_root'])
        self.operator2.start()
        self.operator2.progress.connect(lambda x: self.pbar.setValue(x))
        self.operator2.error_message.connect(lambda x: self.message.append(x))
        self.operator2.done.connect(lambda x: self.delayed_thread_finished())

    @delayed_thread_check_decorator(action_name='OCR for single image')
    def ocr_single(self):
        if not self.config['project_root']:
            self.message.append('Project is not defined.')
            self.delayed_thread_finished()
            return

        fp, _ = QFileDialog.getOpenFileName(self, filter='Images (*.png *.jpeg *.jpg)')
        if not (fp and os.path.exists(fp)):
            self.message.append('The image to recognize does not exist.')
            self.delayed_thread_finished()
            return
        self.input_anytype_displayed.setText(fp)

        self.operator2 = FileOCR(fp, self.config['project_root'])
        self.operator2.start()
        self.operator2.progress.connect(lambda x: self.pbar.setValue(x))
        self.operator2.error_message.connect(lambda x: self.message.append(x))
        self.operator2.done.connect(lambda x: self.delayed_thread_finished())

    @status_check_decorator(action_name='Open project folder')
    def open_project_root(self):
        if not os.path.exists(self.config['project_root']):
            self.message.append('The project folder does not exist.')
        q_url = QUrl()
        QDesktopServices.openUrl(q_url.fromLocalFile(self.config['project_root']))

    def delayed_thread_finished(self):
        self.busy = False
        self.status.showMessage('Ready.', 0)

    def center(self):
        fg = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        fg.moveCenter(cp)
        self.move(fg.topLeft())

    @status_check_decorator(action_name='Print author\'s information')
    def print_author_info(self):
        author_info = '\n'.join([
            'Author:  cloudy-sfu on github.com',
            'Version: 0.1.4',
            'Models:  deep learning models trained by paddlepaddle, baidu company'
        ])
        self.message.append(author_info)

    # Deceive IDE grammar warning; must be written end of the class.
    status_check_decorator = staticmethod(status_check_decorator)
    delayed_thread_check_decorator = staticmethod(delayed_thread_check_decorator)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myw = MyWindow()
    myw.show()
    sys.exit(app.exec_())
