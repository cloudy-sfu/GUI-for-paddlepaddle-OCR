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

        # self.setFont(QFont('Microsoft YaHei', font_size))
        self.setStyleSheet(f'font-family: "Microsoft YaHei", Calibri, Ubuntu; font-size: {font_size}pt;')
        self.resize(800, 480)
        self.setWindowTitle('Chinese and English OCR')
        self.center()

        self.operator1 = None
        self.operator2 = None
        self.busy = False
        self.output_folder = ''
        self.input_folder = ''
        self.input_file = ''
        self.input_clipboard = ''
        self.launch_action = None

        self.create_menu_bar()

        self.output_folder_displayed = QLabel(self)
        self.output_folder_displayed.setText(self.output_folder)
        self.input_anytype_displayed = QLabel(self)
        self.pbar = QProgressBar(self)
        self.message = QTextEdit(self)

        main_part = QWidget(self)
        main_layout = QFormLayout(main_part)
        main_layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        main_layout.addRow('Project:', self.output_folder_displayed)
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
        new_project.triggered.connect(self.define_output_folder)
        new_project.setShortcut('Ctrl+T')

        close_program = QAction('&Exit', self)
        close_program.triggered.connect(self.close)
        close_program.setShortcut('Alt+F4')

        open_folder = QAction('Open f&older ...', self)
        open_folder.triggered.connect(self.define_input_folder)

        open_file = QAction('Open f&ile ...', self)
        open_file.triggered.connect(self.define_input_file)

        paste_from_clipboard = QAction('&Paste from clipboard', self)
        paste_from_clipboard.triggered.connect(self.define_input_clipboard)
        paste_from_clipboard.setShortcut('Ctrl+Shift+V')

        close_input_anytype_action = QAction('&Close all', self)
        close_input_anytype_action.triggered.connect(self.close_input_anytype)
        close_input_anytype_action.setShortcut('Ctrl+W')

        file_menu = QMenu('&File', self)
        file_menu.addActions([new_project, open_folder, open_file, paste_from_clipboard, close_input_anytype_action,
                              close_program])
        menubar.addMenu(file_menu)

        # Run menu
        self.launch_action = QAction('&Launch', self)
        # self.launch_action.triggered.connect is dynamic.

        run_menu = QMenu('&Run', self)
        run_menu.addActions([self.launch_action])
        menubar.addMenu(run_menu)

        # View menu
        open_project_root_action = QAction('&Open project folder ...', self)
        open_project_root_action.triggered.connect(self.open_project_root)

        view_menu = QMenu('&View', self)
        view_menu.addActions([open_project_root_action])
        menubar.addMenu(view_menu)

        # About menu
        about_the_author = QAction('About the auth&or', self)
        about_the_author.triggered.connect(self.print_author_info)

        about_menu = QMenu('&About', self)
        about_menu.addActions([about_the_author])
        menubar.addMenu(about_menu)

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

    @status_check_decorator(action_name='Define project')
    def define_output_folder(self):
        self.output_folder = QFileDialog.getExistingDirectory(self)
        self.output_folder_displayed.setText(self.output_folder)

    @status_check_decorator(action_name='Open folder')
    def define_input_folder(self):
        fp = QFileDialog.getExistingDirectory(self)
        if fp:
            self.input_folder = fp
            try:
                self.launch_action.disconnect()
            except TypeError:
                pass
            self.launch_action.triggered.connect(self.ocr_batch)
            self.input_anytype_displayed.setText(self.input_folder)

    @status_check_decorator(action_name='Copy from clipboard')
    def define_input_clipboard(self):
        if not self.output_folder:
            self.message.append('Project is not defined.')
            return
        clipboard_saved_path = os.path.join(self.output_folder, 'origin_clipboard')
        if not os.path.exists(clipboard_saved_path):
            os.mkdir(clipboard_saved_path)
        clipboard = QApplication.clipboard()
        picture = clipboard.pixmap()
        input_clipboard = os.path.join(clipboard_saved_path, f'hash_{hash(picture)}.png')
        if picture.save(input_clipboard):
            self.input_clipboard = input_clipboard
            self.message.append(f'Saved successfully at {self.input_clipboard}')
            self.launch_action.triggered.connect(self.ocr_clipboard)
            self.input_anytype_displayed.setText(self.input_clipboard)
        else:
            self.message.append('No picture is detected.')

    @status_check_decorator(action_name='Open file')
    def define_input_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, filter='Images (*.png *.jpeg *.jpg)')
        if fp:
            self.input_file = fp
            try:
                self.launch_action.disconnect()
            except TypeError:
                pass
            self.launch_action.triggered.connect(self.ocr_single)
            self.input_anytype_displayed.setText(self.input_file)

    @delayed_thread_check_decorator(action_name='OCR for folder')
    def ocr_batch(self):
        if (not os.path.exists(self.input_folder)) or (not os.path.exists(self.output_folder)) or \
                self.input_folder == self.output_folder:
            self.message.append('The project or opened folder does not exist.')
            self.delayed_thread_finished()
            return
        # define the operator as class-level private property, to prevent being destroyed before finished.
        self.operator1 = FolderOCR(self.input_folder, self.output_folder)
        self.operator1.start()
        self.operator1.progress.connect(lambda x: self.pbar.setValue(x))
        self.operator1.error_message.connect(lambda x: self.message.append(x))
        self.operator1.done.connect(lambda x: self.delayed_thread_finished())

    @delayed_thread_check_decorator(action_name='OCR for single image')
    def ocr_single(self):
        if (not os.path.exists(self.input_file)) or (not os.path.exists(self.output_folder)):
            self.message.append('The project or opened folder does not exist.')
            self.delayed_thread_finished()
            return
        self.operator2 = FileOCR(self.input_file, self.output_folder)
        self.operator2.start()
        self.operator2.progress.connect(lambda x: self.pbar.setValue(x))
        self.operator2.error_message.connect(lambda x: self.message.append(x))
        self.operator2.done.connect(lambda x: self.delayed_thread_finished())

    @delayed_thread_check_decorator(action_name='OCR for clipboard saved image')
    def ocr_clipboard(self):
        if (not os.path.exists(self.input_clipboard)) or (not os.path.exists(self.output_folder)):
            self.message.append('The project or opened folder does not exist.')
            self.delayed_thread_finished()
            return
        self.operator2 = FileOCR(self.input_clipboard, self.output_folder)
        self.operator2.start()
        self.operator2.progress.connect(lambda x: self.pbar.setValue(x))
        self.operator2.error_message.connect(lambda x: self.message.append(x))
        self.operator2.done.connect(lambda x: self.delayed_thread_finished())

    @status_check_decorator(action_name='Close')
    def close_input_anytype(self):
        self.input_folder = ''
        self.input_file = ''
        self.input_anytype_displayed.setText('')
        try:
            self.launch_action.disconnect()
        except TypeError:
            pass

    @status_check_decorator(action_name='Open project folder')
    def open_project_root(self):
        if not os.path.exists(self.output_folder):
            self.message.append('The project folder does not exist.')
        q_url = QUrl()
        QDesktopServices.openUrl(q_url.fromLocalFile(self.output_folder))

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
            'Version: 0.1.3'
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
