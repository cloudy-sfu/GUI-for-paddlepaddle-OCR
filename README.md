# GUI for `paddlepaddle` OCR

 The GUI for `paddlepaddle` optical character recognition

![](https://shields.io/badge/OS-Windows_11-navy)
![](https://shields.io/badge/dependencies-Python_3.12-blue)



*Developer tested in Windows 11. Theoretically we didn't introduce Windows 11 specific features. Windows 10 may work but cannot be guaranteed.*

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

Unzip the latest release and run `GUI for paddlepaddle OCR.exe`



## Install (developers)

Create a Python 3.12 virtual environment and activate.

Run the following commands in Windows command prompt.

```
pip install -r requirements.txt
pyinstaller main.spec
```



## Acknowledgment

<u>The copyright of the unmodified content of the following content belongs to the original author and covered by the original license. The copyright of modified content are covered by our license. "modified content" means the minimum modified characters in the file that makes the program's behavior different.</u>

[Inference models](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/paddleocr.py#L72) as `inference_models`

[paddleocr](https://github.com/PaddlePaddle/PaddleOCR) (modified) as `paddleocr`



<!-- https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec/resolve/main/inference.pdiparams -->
