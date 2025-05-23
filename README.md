# Image Text Removal Pipeline

## Introduction

This project provides a framework for removing text from images, with a particular focus on manga and comics. It employs a multi-stage deep learning pipeline that includes text detection, mask generation, and inpainting. A key feature of this framework is its sophisticated module for generating synthetic training data by adding text to clean images. This allows for robust model training even with limited real-world data.

Currently, the project is a work-in-progress. While the core pipeline structure is established, the actual deep learning models and dataset loading classes for training are placeholders. Future development will involve integrating and training specific models for each stage of the pipeline.

## Pipeline Description

The image text removal process is implemented as a sequential multi-stage pipeline. `TextedImage` objects, which store the original image, the image with synthetic text, text masks, and bounding boxes, are passed through and modified by each stage.

**Step 1: Image Loading and Synthetic Text Generation**

*   The `ImageLoader` class is responsible for loading base images. It can either load images from the `datas/images/clear/` directory or generate noise images if insufficient clean images are available or if `use_noise` is enabled in the settings.
*   A core feature is its ability to synthetically render diverse text onto these base images. This process involves:
    *   Selecting from a variety of fonts located in `datas/fonts/`.
    *   Applying random styles (e.g., color, opacity, stroke, shadow).
    *   Performing transformations such as rotation, shearing, and perspective distortion to mimic real-world text appearance.
*   This stage outputs `TextedImage` objects, which encapsulate the original clean image, the synthetically texted image, a binary mask of the text regions, and a list of bounding boxes for individual text instances.

**Step 2: Bounding Box Merging**

*   The `TextedImage.merge_bboxes_with_margin()` method is called to consolidate text regions.
*   It merges nearby or overlapping bounding boxes based on a defined margin, creating larger, unified bounding boxes that encompass related text elements. This is useful for treating closely spaced words or characters as a single unit for subsequent processing.

**Step 3: Model 1 - Text Object BBox Detection**

*   **Purpose:** This model is intended to accurately detect the bounding boxes of text objects within the input image. It refines or replaces the initial bounding boxes generated during synthetic text creation, especially when dealing with real-world images without pre-existing bounding box information.
*   **Current Status:** The current implementation in `models/Model1.py` is a placeholder. A functional text detection model (e.g., CRAFT, DBNet) needs to be integrated here.

**Step 4: Model 2 - Pixel-wise Mask Generation**

*   **Purpose:** Taking the bounding boxes from Model 1 (or the merged synthetic ones), this model's role is to generate a precise pixel-level binary mask that accurately segments the text pixels from the background pixels within those boxes.
*   **Current Status:** `models/Model2.py` is currently a placeholder. A suitable segmentation model (e.g., U-Net based) needs to be implemented for this stage.

**Step 5: Model 3 - Masked Inpainting**

*   **Purpose:** This is the final stage, responsible for "erasing" the text. Using the pixel-wise mask from Model 2, this model inpaints (reconstructs) the image regions where text was present, aiming to fill them with a plausible background that seamlessly blends with the surrounding image content.
*   **Current Status:** `models/Model3.py` is a placeholder. An advanced inpainting model (e.g., generative adversarial network (GAN) based like LaMa or a diffusion model) is required for effective text removal.

## Project Structure

Here's an overview of the important files and directories within the project:

```
.
├── main.py             # Main script to configure and run the pipeline.
├── Pipeline.py         # Orchestrates the different stages of the image processing pipeline.
├── Utils.py            # Contains general utility classes and functions (e.g., PipelineSetting, ImagePolicy).
├── datas/              # Modules and resources related to data handling.
│   ├── ImageLoader.py  # Responsible for loading images and synthetically generating texted images.
│   ├── TextedImage.py  # Class representing an image with its associated text, mask, and bounding boxes.
│   ├── Dataset.py      # Placeholder for PyTorch Dataset classes (MangaDataset1, etc.).
│   ├── Utils.py        # Data-specific utilities (e.g., BBox, Lang).
│   ├── fonts/          # Directory to store .ttf or .otf font files (e.g., NanumGothic.ttf).
│   ├── images/         # Contains subdirectories for input and output images.
│   │   ├── clear/      # Directory for input images without text.
│   │   └── output/     # Example directory for saving processed images.
│   └── checkpoints/    # Directory for saving model checkpoints (e.g., example.pth).
├── models/             # Modules related to the machine learning models.
│   ├── Model1.py       # Placeholder for Text Object BBox Detection Model.
│   ├── Model2.py       # Placeholder for Pixel-wise Mask Generation Model.
│   ├── Model3.py       # Placeholder for Masked Inpainting Model.
│   └── Utils.py        # Model-specific utilities (e.g., ModelMode).
├── .gitignore          # Specifies intentionally untracked files that Git should ignore.
└── README.md           # This file.
```

*   `main.py`: The main entry point for the application. It handles the configuration and execution of the entire image processing pipeline.
*   `Pipeline.py`: Defines the `PipelineMgr` class, which orchestrates the sequence of operations, including data loading, preprocessing, model inference (or training), and output generation.
*   `Utils.py`: Contains general utility classes like `PipelineSetting` (for configuring pipeline parameters) and `ImagePolicy` (for controlling aspects of synthetic image generation).
*   `datas/`: This directory houses all modules and resources related to data.
    *   `datas/ImageLoader.py`: Contains the `ImageLoader` class, which is crucial for loading initial images and, more importantly, for the sophisticated generation of synthetic text on images. This includes applying various text styles, fonts, and transformations.
    *   `datas/TextedImage.py`: Defines the `TextedImage` class, a central data structure that holds the original image, the synthetically texted version, the text mask, and associated bounding boxes. These objects are passed through the pipeline.
    *   `datas/Dataset.py`: Intended to hold PyTorch `Dataset` classes (like the placeholder `MangaDataset1`) for loading and preparing data for model training.
    *   `datas/Utils.py`: Provides data-specific utility classes and enumerations, such as `BBox` for handling bounding box coordinates and `Lang` for language specification in text generation.
    *   `datas/fonts/`: A directory where font files (TrueType `.ttf` or OpenType `.otf`) are stored. These fonts are used by `ImageLoader` to render text.
    *   `datas/images/`: Contains subdirectories for image assets.
        *   `datas/images/clear/`: This is the designated directory for placing input images that are clean (i.e., without any text).
        *   `datas/images/output/`: An example directory where processed images (e.g., images with text removed) can be saved.
    *   `datas/checkpoints/`: Used for storing saved model weights or checkpoints during and after training.
*   `models/`: This directory contains the modules for the different deep learning models in the pipeline.
    *   `models/Model1.py`: A placeholder for the first model in the pipeline, which is intended for Text Object Bounding Box Detection.
    *   `models/Model2.py`: A placeholder for the second model, designed for Pixel-wise Mask Generation.
    *   `models/Model3.py`: A placeholder for the third model, which will perform the Masked Inpainting to remove text.
    *   `models/Utils.py`: Contains model-specific utilities, such as the `ModelMode` enumeration (e.g., TRAIN, INFERENCE, SKIP) to control model behavior.
*   `.gitignore`: A standard Git file that lists files and directories that should be ignored by the version control system (e.g., temporary files, compiled code, large datasets).
*   `README.md`: The file you are currently reading, providing documentation for the project.

## Setup and Dependencies

Follow these steps to set up your project environment.

**1. Prerequisites:**

*   **Python:** This project requires Python 3.8 or newer.

**2. Recommended Setup (Virtual Environment):**

It is highly recommended to use a virtual environment to manage project-specific dependencies, preventing conflicts with your global Python installation.

*   **Using `venv` (standard Python library):**
    ```bash
    # Create a virtual environment (e.g., named .venv)
    python3 -m venv .venv

    # Activate the virtual environment
    # On macOS and Linux:
    source .venv/bin/activate
    # On Windows:
    # .venv\Scripts\activate
    ```

*   **Using `conda`:**
    ```bash
    # Create a new conda environment (e.g., named image-text-removal)
    conda create -n image-text-removal python=3.8

    # Activate the conda environment
    conda activate image-text-removal
    ```

**3. Dependencies:**

The core Python libraries required for this project are:

*   `torch` (PyTorch): The primary deep learning framework.
*   `torchvision`: Provides datasets, model architectures, and image transformations for PyTorch.
*   `Pillow` (PIL): For image manipulation tasks.
*   `numpy`: For numerical operations, especially with arrays.
*   `matplotlib`: For creating visualizations and plotting (used in `TextedImage.visualize`).
*   `tqdm`: For displaying progress bars during iterative processes.

You can install these dependencies using pip:

```bash
pip install torch torchvision Pillow numpy matplotlib tqdm
```

*Note: A `requirements.txt` file is not yet available for this project. It will be added in a future update to simplify dependency management.*

**4. Fonts:**

For the synthetic text generation feature (`ImageLoader`), you will need font files (TrueType `.ttf` or OpenType `.otf`).

*   Place any font files you wish to use into the `datas/fonts/` directory.
*   The project includes `NanumGothic.ttf` as an example font. You can add other fonts to this directory. The `ImageLoader` will randomly select from any available fonts in this location.

## Usage (Running the Pipeline)

The main script for running the image processing pipeline is `main.py`. You can execute it from the root directory of the project.

**1. Basic Execution:**

```bash
python main.py [options]
```

Replace `[options]` with any of the command-line arguments described below to customize the pipeline's behavior.

**2. Command-Line Arguments:**

The following arguments can be used to configure the pipeline:

*   `--model1 {skip,train,inference}`: Sets the mode for Model 1 (Text Object BBox Detection).
    *   `skip` (default): Skips this model.
    *   `train`: Runs the training process for Model 1.
    *   `inference`: Runs Model 1 for inference.
*   `--model2 {skip,train,inference}`: Sets the mode for Model 2 (Pixel-wise Mask Generation).
    *   `skip` (default): Skips this model.
    *   `train`: Runs the training process for Model 2.
    *   `inference`: Runs Model 2 for inference.
*   `--model3 {skip,train,inference}`: Sets the mode for Model 3 (Masked Inpainting).
    *   `skip` (default): Skips this model.
    *   `train`: Runs the training process for Model 3.
    *   `inference`: Runs Model 3 for inference.
*   `--use_amp`: If specified, enables Automatic Mixed Precision (AMP) for potentially faster training and inference with reduced memory usage, if supported by the hardware.
*   `--num_workers INT`: Sets the number of worker processes for data loading. Default: `0` (data is loaded in the main process).
*   `--num_images INT`: Specifies the total number of images to load/generate for the pipeline. Default: `128`.
*   `--use_noise`: If specified, noise images will be generated and used if the number of clear images found in `datas/images/clear/` is less than `--num_images`.
*   `--margin INT`: Defines the margin (in pixels) to be used for bounding box merging and cropping operations. Default: `4`.
*   `--max_objects INT`: Sets the maximum number of objects (e.g., text bounding boxes) to consider. Default: `1024`. (This is a parameter in `PipelineSetting`.)
*   `--epochs INT`: Specifies the number of training epochs to run. Default: `100`.
*   `--batch_size INT`: Sets the batch size for model training. Default: `4`.
*   `--lr FLOAT`: Sets the learning rate for the optimizer during training. Default: `0.001`.
*   `--weight_decay FLOAT`: Sets the weight decay value for the optimizer. Default: `3e-4`.
*   `--vis_interval INT`: Specifies the interval (in epochs or iterations, context-dependent) at which visualizations should be generated during training. Default: `1`.
*   `--ckpt_interval INT`: Specifies the interval (in epochs) at which model checkpoints should be saved during training. Default: `5`.

**3. Examples:**

Here are a few examples of how to run the pipeline:

*   **Running inference with all three models on 10 images:**
    ```bash
    python main.py --model1 inference --model2 inference --model3 inference --num_images 10
    ```

*   **Training Model 1 for 50 epochs with a batch size of 8 (and skipping other models):**
    ```bash
    python main.py --model1 train --model2 skip --model3 skip --epochs 50 --batch_size 8
    ```

*   **Generating 5 synthetically texted images and visualizing them (without running any models):**
    This involves loading images, applying synthetic text, and then the pipeline finishes. The visualization would typically be part of the `ImageLoader` or `TextedImage` if explicitly saved. For this example, we assume the default behavior might save intermediate results or one would add custom saving/visualization.
    ```bash
    python main.py --num_images 5
    ```
    *(Note: To see the output images, you might need to check the `datas/images/output/` directory or implement specific visualization steps if not already present for this mode.)*

*Important Note: The training functionalities (`--model<1/2/3> train`) are currently based on placeholder model implementations (`Model1.py`, `Model2.py`, `Model3.py`). To perform actual model training, these placeholder scripts need to be replaced with functional deep learning model architectures and training loops.*

## Input and Output

This section details the data the pipeline expects and what it produces.

**1. Input:**

*   **Primary Input Images:**
    *   The pipeline processes standard image files (e.g., PNG, JPG, JPEG).
    *   These images should be placed in the `datas/images/clear/` directory. These are considered the "clean" base images onto which synthetic text will be rendered, or which will be processed by the text removal models.
*   **Alternative Noise Input:**
    *   If the `ImageLoader` does not find a sufficient number of images in `datas/images/clear/` to meet the `--num_images` argument, or if the `--use_noise` flag is specified and conditions require it, the pipeline will generate random noise images.
    *   The dimensions of these generated noise images are typically determined by the `model1_input_size` attribute within the `PipelineSetting` class.
*   **Text Input (Synthetic):**
    *   The text applied to the images is **synthetically generated** by the `ImageLoader` module.
    *   There is no direct input of text from files. Instead, `ImageLoader` uses various internal policies (defined in `ImagePolicy`) and randomly selects fonts from the `datas/fonts/` directory to create diverse text content and styles.

**2. Output:**

*   **In-Memory Processing:**
    *   The core of the pipeline involves the manipulation of `TextedImage` objects in memory. These objects carry the original image, the synthetically texted image, the text mask, and bounding boxes. All transformations, including synthetic text addition, and any (currently placeholder) model predictions or inpainting, occur on these in-memory objects.
*   **Visualization and Saving Images:**
    *   The `TextedImage` class provides a `visualize(dir="path/to/output", filename="name.png")` method. This method can be called to save a composite image that typically shows a side-by-side view of the original image, the texted image (with bounding boxes drawn), and the binary text mask. This is primarily intended for debugging, visualization of intermediate steps, or individual inspection.
    *   The `datas/images/output/` directory serves as an example location where such visualizations or other processed images might be saved. The `example.png` file in this directory is indicative of this type of output.
*   **Pipeline Output (Final Images):**
    *   The main `Pipeline.py` script, in its current state, **does not include explicit steps to automatically save the final processed image** (e.g., the inpainted image after Model 3) to disk at the end of a full pipeline run.
    *   Saving the final result (e.g., the `orig` attribute of the `TextedImage` object after the inpainting model has modified it) would be a feature to implement in future development.
*   **Model Checkpoints:**
    *   If any of the models (Model1, Model2, or Model3) are trained (using the `train` mode), the training process will save model checkpoints (weights and optimizer states).
    *   These checkpoints are stored in the `datas/checkpoints/` directory. The `example.pth` file is an illustration of a saved checkpoint.

## Future Work / To-Do

This project is actively being developed. Here's a list of planned features and potential improvements:

*   **Implement Core Models:**
    *   Develop and integrate the actual neural network architectures for:
        *   `Model1` (Text Object BBox Detection): Replace the placeholder with a functional model like CRAFT, DBNet, or similar.
        *   `Model2` (Pixel-wise Mask Generation): Implement a segmentation model (e.g., U-Net based) for accurate text masking.
        *   `Model3` (Masked Inpainting): Integrate an advanced inpainting model (e.g., GAN-based like LaMa, or a diffusion model) for high-quality text removal.
*   **Full Dataset Integration:**
    *   Complete the implementation of `MangaDataset1`, `MangaDataset2`, and `MangaDataset3` in `datas/Dataset.py`. This will enable training the models with custom or publicly available datasets, moving beyond purely synthetic data.
*   **Output Saving:**
    *   Add a dedicated feature in `Pipeline.py` to automatically save the final processed images (i.e., the inpainted output from Model 3) to a user-specified output directory.
*   **Evaluation Metrics:**
    *   Incorporate standard evaluation metrics to quantitatively assess the performance of each model stage. This includes:
        *   Intersection over Union (IoU) for text detection (Model 1) and mask generation (Model 2).
        *   Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) for evaluating inpainting quality (Model 3).
*   **Requirements File:**
    *   Create a `requirements.txt` file to list all Python dependencies with their specific versions, simplifying environment setup and ensuring reproducibility.
*   **Configuration File Support:**
    *   Introduce support for a configuration file (e.g., using YAML or JSON format) to manage `PipelineSetting`, `ImagePolicy`, and model-specific parameters. This would offer a more organized alternative to using numerous command-line arguments.
*   **Advanced Text Rendering in `ImageLoader`:**
    *   Enhance the synthetic text generation capabilities:
        *   Support for curved or arced text.
        *   More diverse and complex font styling options and visual effects (e.g., gradient fills, complex outlines, perspective text).
        *   Improved simulation of text degradation or noise.
*   **Pre-trained Model Weights:**
    *   Once the core models are implemented and trained, provide pre-trained weights to allow users to quickly use the pipeline for inference without needing to train the models from scratch.
*   **Unit Tests:**
    *   Develop a suite of unit tests for critical components like `ImageLoader`, `TextedImage` methods, and model utilities to ensure code reliability and facilitate easier refactoring.
*   **Documentation Expansion:**
    *   Provide more in-depth documentation for developers, including details on model architectures, training procedures, and how to extend the pipeline or integrate new models.
