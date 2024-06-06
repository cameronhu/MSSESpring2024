import json
import pickle
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import Dict, List
from pathlib import Path


# Constants and classes
######################################################################################

DISASTER_ENCODING = {"hurricane-matthew": 0, "midwest-flooding":1, "socal-fire":2}
ImageInfo = namedtuple("ImageInfo", 'idx, path, disaster, severity')

#####################################################################################

def orient_landscape(img):

    if img.shape[0] < img.shape[1]:
        return img.swapaxes(0, 1)
    else:
        return img

def preprocess_data(config_path: Path, output_root: Path,
                    disaster_encoding: Dict=DISASTER_ENCODING,
                    transforms=List[lambda img: img],
                   ) -> List[ImageInfo]:
    """
    Parse all the images from archive, process images, cache to file, return a list of image info

    Image info is in the form of ImageInfo which has the overall image idx, path to image file, disaster, and severity

    Parameters:
        config_path (Path): path to config file containing "data_dir" field
        output_root (Path): directory to place output. Will be created if needed
        disaster_encoding (Dict): mapping of disaster name to one hot index
        transforms (callable): transformation function(s) to call on each image to cache

    Returns:
        List[ImageInfo]: list of all image info
    """

    output_root.mkdir(exist_ok=True, parents=True)

    with open(config_path, "r") as c:
        config = json.load(c)
    data_dir = Path(config["data_dir"])

    img_idx = 0
    image_data = []
        
    disaster_dirs = data_dir.glob("*/")
    for disaster_dir in disaster_dirs:
        d = disaster_encoding[disaster_dir.name]

        images = np.load(disaster_dir / "train_images.npz", allow_pickle=True)             
        labels = np.load(disaster_dir / "train_labels.npy", allow_pickle=True)

        for i in range(len(labels)):
            severity = labels[i]
            img = images[f"image_{i}"]
            
            for transformation in transforms:
                img = transformation(img)
            img_path = output_root / f"{img_idx}_{d}_{severity}.npy"
            np.save(img_path, img)

            image_data.append(ImageInfo(img_idx, img_path, d, severity))
            img_idx += 1

    with open(output_root / "scaled_image_data.pkl", "wb") as pkl:
        pickle.dump(image_data, pkl)

    return image_data


def load_images(images_path):
    """
    Load images from a specified .npz file.

    Parameters:
    - images_path (str): The file path to the .npz file containing the images.

    Returns:
    - images (list): A list of numpy arrays, each representing an image loaded from the .npz file.
    """
    data = np.load(images_path, allow_pickle=True)
    images = [data[f"image_{i}"] for i in range(len(data.files))]
    return images

def load_labels(labels_path):
    """
    Load labels from a specified .npy file.

    Parameters:
    - labels_path (str): The file path to the .npy file containing the labels.

    Returns:
    - labels (numpy.ndarray): An array of labels loaded from the .npy file.
    """
    labels = np.load(labels_path, allow_pickle=True)
    return labels

def get_images(data_dir, disaster, split="train"):
    """
    Load images from a specified disaster dataset split.

    Args:
        data_dir (str): The directory where the dataset is stored.
        disaster (str): The disaster type of the dataset.
        split    (str): The train or test split (default train).

    Returns:
        list: A list of images (as numpy arrays) from the specified dataset split.
    """
    images_path = os.path.join(data_dir, disaster, f"{split}_images.npz")
    return load_images(images_path)


def get_labels(data_dir, disaster, split="train"):
    """
    Load labels for a specified disaster dataset split.

    Args:
        data_dir (str): The directory where the dataset is stored.
        disaster (str): The disaster type of the dataset.
        split    (str): The train or test split (default train).

    Returns:
        ndarray: The labels for the images in the specified dataset split.
    """
    labels_path = os.path.join(data_dir, disaster, f"{split}_labels.npy")
    return load_labels(labels_path)


def convert_dtype(images, dtype=np.float32):
    """
    Convert the data type of a collection of images.

    Args:
        images (list or dict): The images to convert, either as a list or dictionary of numpy arrays.
        dtype (data-type): The target data type for the images. Defaults to np.float32.

    Returns:
        The converted collection of images, in the same format (list or dict) as the input.
    """
    if isinstance(images, dict):
        return {k: v.astype(dtype) for k, v in images.items()}
    elif isinstance(images, list):
        return [img.astype(dtype) for img in images]
    else:
        raise TypeError("Unsupported type for images. Expected list or dict.")


def plot_label_distribution(labels, ax=None, title="Label Distribution"):
    """
    Plot the distribution of labels.

    Args:
        labels (ndarray): An array of labels to plot the distribution of.
        ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot.
                                             If None, a new figure and axis are created.
        title (str, optional): The title for the plot. Defaults to "Label Distribution".
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True  # Flag indicating a figure was created within this function
    else:
        created_fig = False

    sns.countplot(x=labels, ax=ax, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")

    if created_fig:
        plt.tight_layout()
        plt.show()
