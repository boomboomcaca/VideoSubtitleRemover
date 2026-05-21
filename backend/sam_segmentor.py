import os
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Add root folder and sam folder to path so import segment_anything works
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM_DIR = os.path.join(ROOT_DIR, "sam")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SAM_DIR not in sys.path:
    sys.path.insert(0, SAM_DIR)

class SAMSegmentor:
    """Wrapper class for Segment Anything Model (SAM) used to generate precise masks from clicks."""
    
    def __init__(self, model_type: str = "vit_b", checkpoint_path: str = None, device: str = None):
        self.model_type = model_type
        if checkpoint_path is None:
            self.checkpoint_path = os.path.join(ROOT_DIR, "models", "sam_vit_b_01ec64.pth")
        else:
            self.checkpoint_path = checkpoint_path
            
        self.device = device
        self._predictor = None
        self._image_set = False
        self._current_image = None

    def _lazy_load(self):
        """Lazily loads the SAM model to prevent app startup overhead."""
        if self._predictor is None:
            logger.info("Initializing SAM model...")
            try:
                # Import from Segment Anything
                from segment_anything import sam_model_registry, SamPredictor
            except ImportError:
                # Retry alternative import if paths differ
                try:
                    from sam.segment_anything import sam_model_registry, SamPredictor
                except ImportError as e:
                    logger.error("Failed to import segment_anything. Ensure the 'sam' folder is present.")
                    raise e

            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"SAM checkpoint not found at {self.checkpoint_path}")

            # Auto-detect device if not explicitly set
            if self.device is None:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                else:
                    self.device = "cpu"
            
            logger.info(f"Loading SAM ({self.model_type}) checkpoint onto {self.device}...")
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            self._predictor = SamPredictor(sam)
            logger.info("SAM model loaded successfully.")

    def set_image(self, image: np.ndarray):
        """Set the reference image and precompute its embedding."""
        self._lazy_load()
        # Convert to RGB if in BGR (Pillow RGB is expected by SAM)
        self._predictor.set_image(image)
        self._image_set = True
        self._current_image = image

    def segment_at_point(self, x: int, y: int) -> np.ndarray:
        """
        Runs SAM on the image given a single positive click at (x, y).
        
        Args:
            x: column index (width coordinate)
            y: row index (height coordinate)
            
        Returns:
            binary_mask: np.ndarray of shape (H, W) where True values indicate the segmented object.
        """
        if not self._image_set:
            raise ValueError("Must call set_image() before segmenting.")
            
        import torch
        # Format the point prompt
        input_point = np.array([[x, y]], dtype=np.float32)
        input_label = np.array([1], dtype=np.int32) # 1 represents positive click
        
        with torch.no_grad():
            masks, scores, logits = self._predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
        # Select the mask with the highest IoU prediction score
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx] # boolean ndarray of shape (H, W)
        
        logger.debug(f"SAM prediction completed. Best mask index: {best_idx}, score: {scores[best_idx]:.4f}")
        return best_mask
