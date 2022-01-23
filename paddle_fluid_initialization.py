import os
from paddle.fluid.core_avx import create_paddle_predictor, AnalysisConfig


def set_config(pretrained_model_path):
    """
    predictor config path
    """
    model_file_path = os.path.join(pretrained_model_path, 'model')
    params_file_path = os.path.join(pretrained_model_path, 'params')

    config = AnalysisConfig(model_file_path, params_file_path)
    config.disable_gpu()

    config.disable_glog_info()
    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.switch_use_feed_fetch_ops(False)

    predictor = create_paddle_predictor(config)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_tensor(output_name)
        output_tensors.append(output_tensor)

    return predictor, input_tensor, output_tensors
