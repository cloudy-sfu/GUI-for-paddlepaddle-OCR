# GUI for `paddlepaddle` OCR

 The GUI for `paddlepaddle` OCR

![](https://shields.io/badge/OS-Windows_10_64--bit-lightgray)
![](https://shields.io/badge/dependencies-Python_3.8-blue)
![](https://shields.io/badge/languages-zh,_en-pink)

## Introduction

A Windows GUI to perform optical character recognition, using `paddlepaddle` OCR models. With this program, users can recognize text included in images from both the clipboard and the file system.

Currently, support recognizing both Chinese and Latin characters from  `*.png` and `*.jpg` images.

<details>
 <summary><b>Screenshot</b></summary>
 <img src="assets/image-20230612161702728.png" alt="screenshot">
</details>


## Usage

**Release:**

1. Download and unzip the latest release.
2. Run `GUI-for-paddlepaddle-OCR.exe`.

**Compile from source code:**

Assume the current directory in terminal is the root directory of this program. Denote the virtual environment of Python is defined in Line 1 in `fix_env_win.bat`. By default, I set `VENV=.\venv` which means the virtual environment is installed at `VENV/` relative to the root directory of this program. This variable can be modified according to the user's environment.

Run the following commands in terminal:

```
pip install -r requirements.txt
fix_env_win.bat
pyinstaller main.spec
```

## Acknowledgement

[Inference models](https://github.com/PaddlePaddle/PaddleOCR)  

Licenses of this program don't cover `inference_models/` directory.
