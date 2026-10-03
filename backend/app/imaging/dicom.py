import io
import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np
from PIL import Image

def is_dicom(file_bytes: bytes) -> bool:
    """Check if the byte stream is a valid DICOM file."""
    if len(file_bytes) < 132:
        return False
    return file_bytes[128:132] == b"DICM"

def read_dicom_pixels(file_bytes: bytes) -> np.ndarray:
    """Safely extract pixel data from a DICOM byte stream, ignoring sensitive PHI."""
    try:
        # Stop before reading pixel data to check modality if we wanted, 
        # but here we just read it fully to extract pixels
        dataset = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        
        if not hasattr(dataset, "pixel_array"):
            raise ValueError("DICOM file does not contain pixel data.")
            
        pixels = dataset.pixel_array
        
        # Handle Photometric Interpretation
        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            # Invert so that higher values = brighter (white bones)
            pixels = np.max(pixels) - pixels
            
        return pixels
    except InvalidDicomError as e:
        raise ValueError(f"Invalid DICOM file: {str(e)}")

def read_standard_image(file_bytes: bytes) -> np.ndarray:
    """Read a standard PNG/JPEG image."""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("L")  # convert to grayscale
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Invalid image format: {str(e)}")

def extract_image_array(file_bytes: bytes) -> np.ndarray:
    """Entry point to extract a numpy array from an image byte stream."""
    if is_dicom(file_bytes):
        return read_dicom_pixels(file_bytes)
    else:
        return read_standard_image(file_bytes)
