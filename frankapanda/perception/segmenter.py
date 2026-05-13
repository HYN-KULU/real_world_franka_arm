"""
Grounded-SAM2 segmenter: text labels -> per-pixel int label map.

Loads GroundingDINO + SAM2 once and reuses the predictors across calls.
The `segment` method takes a raw (H, W, 3) uint8 RGB array and a
period-separated label string (e.g. "red cup. blue block. lego.") and
returns a (H, W) int64 label map where -1 = background and k = index
into the parsed object list (also returned).

Detection-label -> global-index mapping: GroundingDINO emits one phrase
per detection; we match each phrase against the user-supplied object
list by case-insensitive substring containment. Multiple detections for
the same label are unioned into the same label_idx (later detections
overwrite earlier ones on overlap).
"""

import sys

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

# --------------------------------------------------------------------------- #
# Import path handling: prefer pip-installed sam2/groundingdino; fall back to
# importing them as a namespace package from the local gsam2 submodule.
# --------------------------------------------------------------------------- #
# `gsam2/sam2/*` uses `from sam2....`           -> need `gsam2/` on sys.path
# `gsam2/grounding_dino/groundingdino/*` uses
#   `from gsam2.grounding_dino.groundingdino....` -> need parent of `gsam2/`
# Adding both lets us import without pip-installing the packages themselves.
_REPO_ROOT = "/home/ksaha/Research/ModelBasedPlanning/real_world_visual_planning"
_GSAM2_DIR = f"{_REPO_ROOT}/gsam2"
for _p in (_REPO_ROOT, _GSAM2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sam2.build_sam import build_sam2                                                # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor                              # noqa: E402
import gsam2.grounding_dino.groundingdino.datasets.transforms as T                    # noqa: E402
from gsam2.grounding_dino.groundingdino.util.inference import load_model, predict     # noqa: E402


# --------------------------------------------------------------------------- #
# Defaults — point at the weights inside the local gsam2 submodule.
# --------------------------------------------------------------------------- #
SAM2_CHECKPOINT = f"{_REPO_ROOT}/gsam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # resolved by sam2's hydra setup
GROUNDING_DINO_CONFIG = f"{_REPO_ROOT}/gsam2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = f"{_REPO_ROOT}/gsam2/gdino_checkpoints/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


def parse_label_list(labels_str: str):
    """`"red cup. blue block. lego"` -> `["red cup", "blue block", "lego"]`."""
    return [s.strip() for s in labels_str.split(".") if s.strip()]


class Segmenter:
    """Grounded-SAM2 wrapper. Loads weights once, segments raw RGB arrays."""

    def __init__(self, device="cuda:0"):
        self.device = device
        print(f"[Segmenter] loading SAM2 from {SAM2_CHECKPOINT}")
        self.sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        print(f"[Segmenter] loading GroundingDINO from {GROUNDING_DINO_CHECKPOINT}")
        self.dino = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=device,
        )
        self._dino_transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print("[Segmenter] ready")

    @torch.no_grad()
    def segment(self, rgb_hw3: np.ndarray, labels_str: str):
        """
        Args:
            rgb_hw3: (H, W, 3) uint8 RGB image (NOT BGR).
            labels_str: period-separated label phrases, e.g. "red cup. blue block.".

        Returns:
            seg_map: (H, W) int64 label map. -1 = background. k = index into object_list.
            object_list: list[str] of parsed labels (also indexable by seg_map values).
        """
        if rgb_hw3.dtype != np.uint8:
            rgb_hw3 = rgb_hw3.astype(np.uint8)
        H, W = rgb_hw3.shape[:2]

        object_list = parse_label_list(labels_str)
        seg_map = -np.ones((H, W), dtype=np.int64)
        if not object_list:
            return seg_map, object_list

        image_pil = Image.fromarray(rgb_hw3)
        image_tensor, _ = self._dino_transform(image_pil, None)

        # SAM2 embedding once per frame.
        self.sam2_predictor.set_image(rgb_hw3)

        # GroundingDINO -> boxes.
        boxes, confidences, det_labels = predict(
            model=self.dino,
            image=image_tensor,
            caption=labels_str,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        if len(boxes) == 0:
            print("[Segmenter] no boxes detected")
            return seg_map, object_list

        boxes = boxes * torch.tensor([W, H, W, H])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # SAM2 -> per-box masks under bf16 autocast.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, _scores, _logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        if masks.ndim == 4:
            masks = masks.squeeze(1)  # (N, H, W)

        # Paint each detection's mask into seg_map at the matched label index.
        for det_idx in range(masks.shape[0]):
            obj_idx = self._match_label_to_object(det_labels[det_idx], object_list)
            if obj_idx < 0:
                continue
            seg_map[masks[det_idx].astype(bool)] = obj_idx

        # Logging summary.
        for k, name in enumerate(object_list):
            n_px = int((seg_map == k).sum())
            print(f"[Segmenter]   label {k} ('{name}'): {n_px} pixels")
        return seg_map, object_list

    @staticmethod
    def _match_label_to_object(det_label: str, object_list):
        det_low = det_label.lower()
        for i, name in enumerate(object_list):
            name_low = name.lower()
            if name_low in det_low or det_low in name_low:
                return i
        return -1
