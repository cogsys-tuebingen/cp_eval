import re
import numpy as np
import cv2

from converter import *

VALID_CONVERTERS = {
    'bitdepth': BitDepthConverter,
    'compression': CompressionLevelSet,
    'colorspace': ColorSpaceConverter,
    'gamma': GammaConverter,
    'distortion': LensDistortionSim,
    'dynamic': DynamicGammaConverter,
    'colorspacereduction': ColorSpaceReduction,
    'downscale': Downscale
}


def _parse_arguments(arg_str: str):
    matches = re.findall(r'([^, ]+)[, ]*', arg_str)
    matches = [s[1:-1] if s[0] == "'" else float(s) for s in matches]

    return matches


def _parse_str(s: str):
    commands = re.findall(r"([a-z]*)\(((?:[^), ]*[, ]*)*)\)[, ]*", s)

    commands = [(command[0].lower(), _parse_arguments(command[1])) for command in commands]

    preprocessing = []

    for command in commands:
        if command[0] not in VALID_CONVERTERS.keys():
            raise Exception(f"Invalid converter specified: {command[0]}")

        preprocessing.append(VALID_CONVERTERS[command[0]](*command[1]))

    return preprocessing


def _parse_preprocessing_command(s: str):
    if s is None:
        preprocessing = []
    else:
        preprocessing = _parse_str(s)

    return preprocessing


def convert_2_sample(img):
    return {
        'img': img,
        'annot': np.array([])
    }


def extract_image(sample):
    return sample['img']

def get_used_jpeg_compression_quality(converter):
    """
        checks whether jpeg compression is part of the compression pipeline.
        if yes the lowest compression quality is return otherwise False
    """
    jpeg_converters = [c for c in converter if isinstance(c, CompressionLevelSet) and c.compression_strategy == 'jpeg']
    if len(jpeg_converters) > 0:
        return int(min([c.to_compression_level for c in jpeg_converters]))
    else:
        return False


def generate_preprocessing_pipelines(train_pipeline_str, validation_pipeline_str=None,
                                     test_pipeline_str=None, only_image_input=False):
    train_pipeline = _parse_preprocessing_command(train_pipeline_str)

    if validation_pipeline_str is not None:
        validation_pipeline = _parse_preprocessing_command(validation_pipeline_str)
    else:
        validation_pipeline = train_pipeline

    if test_pipeline_str is not None:
        test_pipeline = _parse_preprocessing_command(test_pipeline_str)
    else:
        test_pipeline = train_pipeline

    if only_image_input:
        train_pipeline = [convert_2_sample, *train_pipeline, extract_image]
        validation_pipeline = [convert_2_sample, *validation_pipeline, extract_image]
        test_pipeline = [convert_2_sample, *test_pipeline, extract_image]

    return train_pipeline, validation_pipeline, test_pipeline


if __name__ == '__main__':
    preprocessing = generate_preprocessing_pipelines("bitdepth(2, 3), bitdepth(4)")
    preprocessing2 = generate_preprocessing_pipelines("colorspace('RGB'), colorspace('YCrCb')")
    preprocessing3 = generate_preprocessing_pipelines("compression('jpeg', 80)")
    preprocessing4 = generate_preprocessing_pipelines("gamma(2)")
    preprocessing5 = generate_preprocessing_pipelines("distortion(2e-1)")
    preprocessing6 = generate_preprocessing_pipelines("dynamic()")
    preprocessing7 = generate_preprocessing_pipelines("dynamic()", only_image_input=True)

    print(preprocessing)
    print(preprocessing2)
    print(preprocessing3)
    print(preprocessing4)
    print(preprocessing5)
    print(preprocessing6)
    print(preprocessing7)
