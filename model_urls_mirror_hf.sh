# CONSTANT
# -----------------------------
export HF_TOKEN=""
# -----------------------------

curl https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar --output PP-OCRv5_server_det_infer.tar
curl https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar --output Multilingual_PP-OCRv3_det_infer.tar
curl https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar --output PP-OCRv5_server_rec_infer.tar
curl https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv5_mobile_rec_infer.tar --output korean_PP-OCRv5_mobile_rec_infer.tar
curl https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ta_PP-OCRv4_rec_infer.tar --output ta_PP-OCRv4_rec_infer.tar
curl https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/te_PP-OCRv4_rec_infer.tar --output te_PP-OCRv4_rec_infer.tar
curl https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ka_PP-OCRv4_rec_infer.tar --output ka_PP-OCRv4_rec_infer.tar
curl https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv5_mobile_rec_infer.tar --output latin_PP-OCRv5_mobile_rec_infer.tar
curl https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/arabic_PP-OCRv4_rec_infer.tar --output arabic_PP-OCRv4_rec_infer.tar
curl https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/eslav_PP-OCRv5_mobile_rec_infer.tar --output eslav_PP-OCRv5_mobile_rec_infer.tar
curl https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/devanagari_PP-OCRv4_rec_infer.tar --output devanagari_PP-OCRv4_rec_infer.tar
curl https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/th_PP-OCRv5_mobile_rec_infer.tar --output th_PP-OCRv5_mobile_rec_infer.tar
curl https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/el_PP-OCRv5_mobile_rec_infer.tar --output el_PP-OCRv5_mobile_rec_infer.tar
curl https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar --output en_PP-OCRv5_mobile_rec_infer.tar
curl https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar --output PP-LCNet_x1_0_textline_ori_infer.tar
pip install -U "huggingface_hub[cli]"
hf auth login --token $HF_TOKEN --add-to-git-credential
hf upload cloudy-sfu/Paddle-OCR .
