# GUI for `paddlepaddle` OCR

 The GUI for `paddlepaddle` optical character recognition

![](https://shields.io/badge/OS-Windows_10_64--bit-navy)
![](https://shields.io/badge/dependencies-Python_3.11-blue)

## Introduction

 The GUI for `paddlepaddle` optical character recognition.

Supported images formats: `.png, .jpg`

Supported source: clipboard, single image, folder of multiple images.

Supported languages (characters sets):

- Chinese (simplified) & Latin
- Latin
- Arabic
- Cyrillic
- Devanagari
- Chinese (traditional)
- Korean
- Japan
- Tamil
- Telugu
- Kabardian

![image-20250511161324886](./assets/image-20250511161324886.png)

## Usage

**From release:**

Download and unzip the latest release.

Run `GUI for paddlepaddle OCR.exe`.



**From source code:**

Create a Python 3.11 virtual environment and activate.

Run the following commands in Windows command prompt.

```
pip install -r requirements.txt
pyinstaller main.spec
```



## Acknowledgment

<u>The copyright of the unmodified content of the following content belongs to the original author and covered by the original license. The copyright of modified content are covered by our license. "modified content" means the minimum modified characters in the file that makes the program's behavior different.</u>

[Inference models](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/paddleocr.py#L72) as `inference_models`

[Intel oneAPI math kernel library 2019](https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-math-kernel-library-release-notes-and-new-features.html#inpage-nav-1) as `mklml`

[paddleocr](https://github.com/PaddlePaddle/PaddleOCR) (modified) as `paddleocr`


