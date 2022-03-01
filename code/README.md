# Camera Parameters Evaluation
This is the repository is part of the official implementation of the paper 'Comprehensive Analysis of the Object Detection Pipeline'. It provides the code to generate image variations with custom converters with regard to the factors of influence discussed in the paper.

## Example
1. Install the requirements.txt
2. Execute "python3 apply_to_imgs.py [input path] [output path] [target converter]"
3. Output can be found in output path. Images have been converted with chosen convert.


## Folder structure
 - 'apply_to_imgs.py' Provides interactive visualization of the included factors of influence
 - 'preprocessor_factor.py' Provides a factory-like wrapper for all image converters
 - 'converter' Contains the code to for each convert representing one factor of influence
  
## Convert Overview
### Quantisation
Convert to reduce the bit depth of an image to reduce file size.

### Compression
Convert to apply common JPEG compression for fast file size reduction of an image. See ```converter/compression_level.py```

### Resolution
Convert to rescale the image to a larger or smaller resolution. See ```converter/downscale.py```

### Color Model
Convert to change the color model of the image representation. See ```converter/color_space.py```

### Image Distortion
Convert to introduces a small image distortion to the image and bounding boxes. See ```converter/lens_distortion_sim.py```

### Additional Channels / Color Space Reduction
Convert to map multi-spectral color space BGRNE to standard RGB. See ```converter/color_space_reduction.py```

### Gamma Correction
Convert to adjust the gamma level of an image to improve contrast in the image. See ```converter/gamma_correction.py``` and ```converter/dynamic_gamma_correction.py```