# GUI for "paddlepaddle" OCR

 The GUI for "paddlepaddle" OCR, standalone version

![](https://shields.io/badge/OS-Windows%2010%2064--bit-lightgray.svg)
![](https://shields.io/badge/dependencies-Python%203.9-blue.svg)
![](https://shields.io/badge/language-Chinese,%20English-pink.svg)

## Introduction

A Windows GUI to perform optical character recognition, using "paddlepaddle" 
OCR models. With this program, users can recognize text included in images 
from both the clipboard and the file system.

Currently, support

- Language: Chinese, English
- Format of images: `*.png`, `*.jpg`

<img src="https://user-images.githubusercontent.com/41314224/152323922-6b36c258-8908-4ba0-a50b-b21e1d069754.png"      width="400px" alt="screenshot">

## Citation

All used packages are listed in `requirements.txt`. 

Specially, `paddlepaddle` are disassembled and separately used. The OCR models
(in `inference_models` folder included in source code and released program)
and `*.dll` binary files (in `paddle/libs` folder included in released program)
are provided by "paddlepaddle". Therefore, **ONLY IF YOU TRUST** 
"paddlepaddle", can you use files (both source code and released program)
in this repository.

## Usage

Download the latest release of this repository, unzip and run the shortcut of `ocr_win64.exe`.

### 1. Compiling

**Windows 10 64-bit**

(1) Make the root folder of Python external library, such as`venv\Lib\site-packages` or `~/.conda/envs/.../Lib/site-packages`, as the current folder.

(2) Modify line 16 of `paddle/fluid/proto/pass_desc_pb2.py`

Replace

```python
import framework_pb2 as framework__pb2
```

with

```python
from . import framework_pb2 as framework__pb2
```

(3) Modify line 39-62 of `paddle/dataset/image.py`

Replace

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

with

```python
import cv2
```

(4) Make the project root as the current folder.

(5) Run

```bash
pyinstaller main.spec 
Xcopy /E /I inference_model dist\main\inference_model
```
