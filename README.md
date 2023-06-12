# GUI for "paddlepaddle" OCR

 The GUI for "paddlepaddle" OCR, standalone version

![](https://shields.io/badge/OS-Windows%2010%2064--bit-lightgray.svg)
![](https://shields.io/badge/dependencies-Python%203.10-blue.svg)
![](https://shields.io/badge/languages-zh,%20en-pink.svg)

## Introduction

A Windows GUI to perform optical character recognition, using "paddlepaddle" OCR models. With this program, users can recognize text included in images from both the clipboard and the file system.

Currently, support recognizing both Chinese and Latin characters from  `*.png` and `*.jpg` images.

<details>
 <summary><b>Screenshot</b></summary>
 <img src="https://user-images.githubusercontent.com/41314224/152323922-6b36c258-8908-4ba0-a50b-b21e1d069754.png" alt="screenshot">
</details>

## Usage

**Release:**

1. Download and unzip the latest release.
2. Run `GUI-for-paddlepaddle-OCR.exe`.

**Compile from source code:**

1. Make the root folder of Python external library, such as    `venv\Lib\site-packages` (in Windows) or `~/.conda/envs/.../Lib/site-packages`  (in Linux), as the current folder in terminal.

2. Modify line 16 of `paddle/fluid/proto/pass_desc_pb2.py`: replace   `import framework_pb2 as framework__pb2` with   `from . import framework_pb2 as framework__pb2`.

3. Modify line 39-62 of `paddle/dataset/image.py`: replace
    ```python
    if six.PY3:
        import subprocess
        import sys
        import os
        interpreter = sys.executable
        # Note(zhouwei): if use Python/C 'PyRun_SimpleString', 'sys.executable'
        # will be the C++ execubable on Windows
        if sys.platform == 'win32' and 'python.exe' not in interpreter:
            interpreter = sys.exec_prefix + os.sep + 'python.exe'
        import_cv2_proc = subprocess.Popen(
            [interpreter, "-c", "import cv2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        out, err = import_cv2_proc.communicate()
        retcode = import_cv2_proc.poll()
        if retcode != 0:
            cv2 = None
        else:
            import cv2
    else:
        try:
            import cv2
        except ImportError:
            cv2 = None
    ```
    with `import cv2`.

4. Comment line 35-46 of `_pyinstaller_hooks_contrib\hooks\stdhooks\hook-shapely.py` as follows.
    ```python
        # original_path = os.environ['PATH']
        # try:
        #     os.environ['PATH'] = os.pathsep.join(lib_paths)
        #     dll_path = find_library('geos_c')
        # finally:
        #     os.environ['PATH'] = original_path
        # if dll_path is None:
        #     raise SystemExit(
        #         "Error: geos_c.dll not found, required by hook-shapely.py.\n"
        #         "Please check your installation or provide a pull request to "
        #         "PyInstaller to update hook-shapely.py.")
        # binaries += [(dll_path, '.')]
    ```

5. Make the project root as the current folder in terminal.

6. Run `pyinstaller main.spec` in terminal.

## Acknowledgement

All used packages are listed in `requirements.txt`.

Specifically, the files in `inference_models` are provided by `paddlepaddle`. See the license at https://github.com/PaddlePaddle/PaddleOCR before to copy, share, redistribute, modify them.
