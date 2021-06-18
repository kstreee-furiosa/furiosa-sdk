__all__ = ["debug_ptq"]

import importlib
from typing import List, Dict

import onnx
import onnxruntime as ort
import numpy as np

import furiosa_sdk_quantizer
from furiosa_sdk_quantizer.frontend.onnx import optimize_model, quantize, quantizer, calibrate
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model


utils = importlib.import_module('furiosa').utils
__version__ = utils.get_sdk_version(__name__)


def debug_ptq(
        model: onnx.ModelProto,
        validation_dataset: List[Dict[str, np.ndarray]],
        quantized_model: onnx.ModelProto = None,
        calibration_dataset: List[Dict[str, np.ndarray]] = None,
        per_channel: bool = True,
        threshold: float = 1.0,
) -> onnx.ModelProto:
    """Debug post-training-quantizes an ONNX model with a calibration dataset.

    Args:
        model: An ONNX model to quantize.
        validation_dataset: A validation dataset.
        quantized_model: A quantized model.
        calibration_dataset: A calibration dataset, required when no quantized_model given.
        per_channel: If per_channel is True, Conv's filters are
          per-channel quantized. Otherwise, they are per-tensor
          quantized.
        threshold: A threshold as counting 'different element' between tensors
    Returns:
        An ONNX model post-training-quantized with the calibration
        dataset.
    """
    ort.set_default_logger_severity(3)

    check_model(model)
    model = optimize_model(model)
    if quantized_model is None:
        assert (calibration_dataset is not None, "Calibration dataset is required when no quantized_model given")
        ranges = calibrate(model, calibration_dataset)
        quantized_model = quantize(model, per_channel, True, quantizer.QuantizationMode.fake, ranges)

    vi_map = {vi.name: vi for vi in model.graph.value_info}
    quantized_vi_map = {vi.name[:-len('_dequantized')]: vi for vi in
                        filter(lambda vi: vi.name.endswith('_dequantized'), quantized_model.graph.value_info)}
    common_output_activations = list(set(vi_map.keys()) & set(quantized_vi_map.keys()))

    model.graph.output.extend(list(map(lambda activation: vi_map[activation], common_output_activations)))
    quantized_model.graph.output.extend(list(
        map(lambda activation: quantized_vi_map[activation], common_output_activations)))

    model_outputs = []
    session = ort.InferenceSession(model.SerializeToString())
    for inputs in validation_dataset:
        model_outputs.append(session.run(common_output_activations, inputs))
    del session

    quantized_model_outputs = []
    session = ort.InferenceSession(quantized_model.SerializeToString())
    for inputs in validation_dataset:
        quantized_model_outputs.append(session.run(common_output_activations, inputs))
    del session

    for activation, model_output, quantized_model_output in zip(common_output_activations,
                                                                model_outputs,
                                                                quantized_model_outputs):
        diff = np.count_nonzeros(np.absolute(np.subtract(model_output, quantized_model_output)) > threshold)
        print(f'Tensor {activation}: number of different elements where greater than threshold ({threshold}): {diff}')
