# GUI for `paddlepaddle` OCR

 The GUI for `paddlepaddle` OCR

![](https://shields.io/badge/OS-Windows_10_64--bit-lightgray?style=flat-square)
![](https://shields.io/badge/dependencies-Python_3.8-blue?style=flat-square)
![](https://shields.io/badge/languages-zh,_en-pink?style=flat-square)

## Introduction

A Windows GUI to perform optical character recognition using `paddlepaddle` OCR models. With this program, users can recognize text included in images from both the clipboard and the file system.

Support recognizes Chinese and Latin characters from  `*.png` and `*.jpg` images.

<details>
 <summary><b>Screenshot</b></summary>
 <img src="assets/image-20230612161702728.png" alt="screenshot">
</details>


## Usage

**Release:**

1. Download and unzip the latest release.
2. Run `GUI-for-paddlepaddle-OCR.exe`.

**Compile from source code:**

Download `mklml_win_2018.0.3.20180406.zip` from [Microsoft](https://learn.microsoft.com/en-us/cognitive-toolkit/setup-mkl-on-windows), unzip, and put in the program root as directory `mklml_win_2018.0.3.20180406/`. The expected file structure in this directory is shown in `mklml_index`.

Run the following commands in the terminal:

```
pip install -r requirements.txt
pyinstaller main.spec
```

## Acknowledgment

[Inference models](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/ppocr_introduction.md#6-%E6%A8%A1%E5%9E%8B%E5%BA%93)

The license of this program doesn't cover `inference_models/` and `paddleocr/` directory.
