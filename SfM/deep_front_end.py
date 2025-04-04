import sys
import kornia
from abc import ABC, abstractmethod
from torch import nn
from lightglue import LightGlue as LightGlueCore


class ModelWrapper(nn.Module, ABC):
    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config or {}
        self.required_keys = []
        self.configure(self.config)
        sys.stdout.flush()

    def forward(self, batch: dict) -> dict:
        for key in self.required_keys:
            if key not in batch:
                raise KeyError(f"Missing required input: {key}")
        return self.inference(batch)

    @abstractmethod
    def configure(self, config: dict):
        pass

    @abstractmethod
    def inference(self, batch: dict) -> dict:
        pass


class DISK(ModelWrapper):
    def configure(self, config: dict):
        self.params = {
            "weights": config.get("weights", "depth"),
            "max_kpts": config.get("max_keypoints"),
            "nms_size": config.get("nms_window_size", 5),
            "score_thresh": config.get("detection_threshold", 0.0),
            "auto_pad": config.get("pad_if_not_divisible", True),
        }
        self.extractor = kornia.feature.DISK.from_pretrained(self.params["weights"])
        self.required_keys = ["image"]

    def inference(self, batch: dict) -> dict:
        img_tensor = batch["image"]
        results = self.extractor(
            img_tensor,
            n=self.params["max_kpts"],
            window_size=self.params["nms_size"],
            score_threshold=self.params["score_thresh"],
            pad_if_not_divisible=self.params["auto_pad"],
        )

        output_dict = {
            "keypoints": [res.keypoints for res in results],
            "keypoint_scores": [res.detection_scores for res in results],
            "descriptors": [res.descriptors.T for res in results],
        }
        return output_dict
    
class LightGlue(ModelWrapper):
    def configure(self, config: dict):
        self.params = {
            "features": config.get("features", "superpoint"),
            "depth_confidence": config.get("depth_confidence", 0.95),
            "width_confidence": config.get("width_confidence", 0.99),
        }
        self.net = LightGlueCore(self.params["features"],
                                 depth_confidence=self.params["depth_confidence"],
                                 width_confidence=self.params["width_confidence"])

        self.required_keys = [
            "image0", "keypoints0", "descriptors0",
            "image1", "keypoints1", "descriptors1",
        ]

    def inference(self, batch: dict) -> dict:
        # Transpose descriptors to shape [C, N]
        batch["descriptors0"] = batch["descriptors0"].transpose(-1, -2)
        batch["descriptors1"] = batch["descriptors1"].transpose(-1, -2)

        input0 = {k[:-1]: v for k, v in batch.items() if k.endswith("0")}
        input1 = {k[:-1]: v for k, v in batch.items() if k.endswith("1")}

        return self.net({"image0": input0, "image1": input1})