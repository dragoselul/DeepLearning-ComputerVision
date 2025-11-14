import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import dataset as ds
from models.EncDecModel import EncDec
from models.UNetModel import UNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def save_mask(array, path):
    """
    Save a binary segmentation mask to a file.

    Args:
        array: 2D numpy array with 0s and 1s
        path: Output file path
    """
    # array should be a 2D numpy array with 0s and 1s
    # np.unique(array) == [0, 1]
    # len(np.shape(array)) == 2
    im_arr = (array * 255)
    Image.fromarray(np.uint8(im_arr)).save(path)


def load_model(model_path, model_type='encdec', device='cuda'):
    """
    Load a trained segmentation model.

    Args:
        model_path: Path to the saved model (.pth file)
        model_type: Type of model ('encdec', 'unet')
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        Loaded model in evaluation mode
    """
    # Load the entire model (if saved with torch.save(model, path))
    if model_path.endswith('.pth') and os.path.exists(model_path):
        try:
            # Try loading the entire model first
            model = torch.load(model_path, map_location=device)
            print(f"Loaded complete model from {model_path}")
        except:
            # If that fails, load state dict into a new model
            if model_type == 'encdec':
                model = EncDec()
            elif model_type == 'unet':
                model = UNet()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = model.to(device)
    model.eval()  # Set to evaluation mode

    return model


def predict_batch(model, images, device, threshold=0.5):
    """
    Generate predictions for a batch of images.

    Args:
        model: The segmentation model
        images: Batch of input images (tensor)
        device: Device to run inference on
        threshold: Threshold for binary segmentation (default: 0.5)

    Returns:
        Binary predictions as numpy array
    """
    with torch.no_grad():
        images = images.to(device)

        # Forward pass
        outputs = model(images)

        # Apply sigmoid to get probabilities (if outputs are logits)
        probs = torch.sigmoid(outputs)

        # Threshold to get binary predictions
        predictions = (probs > threshold).float()

        # Move to CPU and convert to numpy
        predictions = predictions.cpu().numpy()

    return predictions


def predict_and_save(model_path, test_loader, output_dir, model_type='encdec',
                     device='cuda', threshold=0.5):
    """
    Generate predictions for all test images and save them.

    Args:
        model_path: Path to the trained model
        test_loader: DataLoader for test set
        output_dir: Directory to save predictions
        model_type: Type of model architecture
        device: Device for inference
        threshold: Binary threshold
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, model_type, device)

    print(f"Generating predictions...")
    print(f"Saving predictions to {output_dir}")

    # Get image paths from dataset
    dataset = test_loader.dataset
    image_paths = dataset.image_paths

    image_idx = 0
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        predictions = predict_batch(model, images, device, threshold)

        for i in range(predictions.shape[0]):
            if predictions.ndim == 4:
                mask = predictions[i, 0, :, :]
            else:
                mask = predictions[i, :, :]

            # Get original filename from image path
            original_filename = os.path.basename(image_paths[image_idx])
            # Change extension to .png
            filename = os.path.splitext(original_filename)[0] + '.png'

            output_path = os.path.join(output_dir, filename)
            save_mask(mask, output_path)

            image_idx += 1

    print(f"Saved {image_idx} predictions to {output_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'drive_encdec_model.pth')
    output_dir = os.path.join(script_dir, 'predictions/DRIVE')
    size = 128
    test_transform = transforms.Compose([transforms.Resize((size, size)),
                                         transforms.ToTensor()])
    testset = ds.DRIVE(split="test", transform=test_transform)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)
    predict_and_save(model_path, test_loader, output_dir, model_type='encdec', device='cuda', threshold=0.5)