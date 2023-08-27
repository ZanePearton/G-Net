# G-Net: A TensorFlow-based Generative Adversarial Network

G-Net is an implementation of Generative Adversarial Networks (GAN) using TensorFlow. Designed to generate images of size 32x32 with 3 channels (RGB), G-Net provides real-time training statistics through TensorBoard integration.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Features

- **TensorFlow GAN**: Built with TensorFlow, G-Net offers a robust platform for GAN model development.
  
- **TensorBoard Integration**: Get insights into your model's performance in real-time with TensorBoard logging.
  
- **Customizable Noise Dimension**: Tailor the noise dimension based on your specific requirements.
  
- **Image Visualization**: Track the evolution of the generated images through visualization after regular intervals.

## Requirements

- TensorFlow (2.x recommended)
- numpy
- matplotlib
- pickle

## Usage

1. **Setup**:
   Start by installing all required libraries:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

2. **Dataset Preparation**:
   Use datasets in pickle format that can be reshaped into images of (32, 32, 3). The given function `load_and_preprocess_data` is set up for this purpose.

3. **Training**:
   To initiate model training, run:
   ```bash
   python <filename>.py
   ```
   (Replace `<filename>` with the name of the Python script containing the G-Net code).

4. **Monitor with TensorBoard**:
   Track the training progress visually using TensorBoard:
   ```bash
   tensorboard --logdir logs/
   ```

5. **Tweaking**:
   Adjust the `noise_dim` to modify the noise dimension. For a different number of epochs, change the `epochs` variable.

## License

G-Net is an open-source project under the [MIT License](https://opensource.org/licenses/MIT).

---

**Note**: Ensure you include a license file if you reference it in the README.