# GUI for `paddlepaddle` OCR

 The GUI for `paddlepaddle` optical character recognition

![](https://shields.io/badge/OS-Windows_11-navy)
![](https://shields.io/badge/dependencies-Python_3.12-blue)



*Built and tested on Windows 11. Theoretically not introduced features which are specific for Windows 11. Older Windows versions may work but cannot be guaranteed.*

Supported images formats: `.png, .jpg`

Supported source: clipboard, single image, folder of multiple images.

Supported languages (characters sets):

- Chinese (simplified) & English
- English
- Latin [^1]
- Arabic
- Cyrillic
- Devanagari
- Chinese (traditional)
- Korean
- Japan
- Tamil
- Telugu
- Kabardian

[^1]: English characters set equal to non-space ASCII. Latin misses some punctuations (e.g. no backslash) but has decorated letters (e.g. `Å`).



## Usage

Unzip the latest release and run `GUI for paddlepaddle OCR.exe`

For the first time starting the program, it takes several minutes[^2] before the main window is open and the icon appears in taskbar.

[^2]: Observed and measured in the following hardware level: AMD Ryzen 5 5600U, 16GB DDR4 memory.

Press `Ctrl + L` or select `File -> Switch language` in the navigation bar.

![image-20250923170832482](./assets/image-20250923170832482.png)

Select the language that text in image will be converted to. It can be an option for one language or mixed language, but only support one option instead of multiple.

![image-20250923170235305](./assets/image-20250923170235305.png)

Wait while the program is downloading and initializing the OCR model.

![image-20250923170412332](./assets/image-20250923170412332.png)

>   [!NOTE]
>
>   The user may see connection error. In this case, try the following steps one by one until the initialization is successful. 
>
>   1.   Repeat the "switching language" step and the model will be re-initialized.
>   2.   Ensure network connection to `paddleocr.bj.bcebos.com` [^3] is good.
>   3.   Click `File -> Clear inference models` and repeat step 1, to remove broken files. Alternatively, click "Open inference models folder" and remove broken files manually.
>        ![image-20250923172603201](./assets/image-20250923172603201.png)



[^3]: [Source code](https://github.com/cloudy-sfu/GUI-for-paddlepaddle-OCR/blob/paddleocr-v4/paddleocr/model_urls.json)

Prepare the image(s), from clipboard, as single file, or as a folder to OCR.

![image-20250923172812482](./assets/image-20250923172812482.png)

Read the result in the message box.

The progress only shows recognized images over total number of images, so the progress will be either 0% or 100% if recognizing single image from single file or clipboard.

## Install



### From source code

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
