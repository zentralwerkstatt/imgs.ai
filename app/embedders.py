import numpy as np
import PIL.Image
import torch as t
import torchvision as tv
import torch.nn as nn
import clip


def from_device(tensor):
    return tensor.detach().cpu().numpy()


class Embedder_Raw:

    model = None

    def __init__(self, resolution=32, reducer=None, metrics=["manhattan"]):
        self.resolution = resolution
        self.feature_length = self.resolution * self.resolution * 3
        self.reducer = reducer
        self.metrics = metrics

    def transform(self, img, device="cpu"):
        img = img.resize((self.resolution, self.resolution), PIL.Image.LANCZOS)
        output = np.array(img)
        return output.astype(np.uint8).flatten()


class Embedder_VGG19:

    feature_length = 4096
    model = None
    
    def __init__(self, reducer=None, metrics=["manhattan"]):
        self.reducer = reducer
        self.metrics = metrics

    def transform(self, img, device="cpu"):
        if self.model is None:
            # Construct model only on demand
            self.model = tv.models.vgg19(pretrained=True).to(device)
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier.children())[:5]
            )  # VGG19 fc1
            self.model.eval()
            self.transforms = tv.transforms.Compose(
                [tv.transforms.Resize((224, 224)), tv.transforms.ToTensor()]
            )

        with t.no_grad():
            output = self.model(self.transforms(img).unsqueeze(0).to(device))
            return from_device(output).astype(np.float32).flatten()


class Embedder_CLIP_ViT:

    feature_length = 512
    model = None

    def __init__(self, reducer=None, metrics=["angular"]):
        self.reducer = reducer
        self.metrics = metrics

    def transform(self, img, device="cpu"):
        if self.model is None:
            self.model, self.transforms = clip.load("ViT-B/32", device=device)
            self.model.eval()

        with t.no_grad():
            input_ = self.transforms(img).unsqueeze(0).to(device)
            output = self.model.encode_image(input_)
            output /= output.norm(dim=-1, keepdim=True)
            return from_device(output).astype(np.float32).flatten()


class Embedder_Poses:

    feature_length = 17 * 2
    model = None

    def __init__(self, min_score=0.9, reducer=None, metrics=["manhattan", "angular", "euclidean"]):
        self.min_score = min_score
        self.reducer = reducer
        self.metrics = metrics

    def _normalize_keypoints(self, keypoints, scores):
        keypoints_scaled = np.zeros(self.feature_length) # Return empty array if no poses found
        if keypoints.shape[0] > 0:
            keypoints = keypoints[0]  # Already ranked by score
            score = scores[0].item()
            if self.min_score is None or score > self.min_score:
                # Scale w.r.t exact bounding box
                min_x = min([keypoint[0] for keypoint in keypoints])
                max_x = max([keypoint[0] for keypoint in keypoints])
                min_y = min([keypoint[1] for keypoint in keypoints])
                max_y = max([keypoint[1] for keypoint in keypoints])
                keypoints_scaled = []
                for keypoint in keypoints:
                    scaled_x = (keypoint[0] - min_x) / (max_x - min_x)
                    scaled_y = (keypoint[1] - min_y) / (max_y - min_y)
                    keypoints_scaled.extend([scaled_x, scaled_y])
                keypoints_scaled = np.array(keypoints_scaled)
        return keypoints_scaled

    def transform(self, img, device="cpu"):
        if self.model is None:
            self.model = tv.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device)
            self.model.eval()
            self.transforms = tv.transforms.Compose([tv.transforms.Resize(256), tv.transforms.ToTensor()])

        with t.no_grad():
            output = self.model(self.transforms(img).unsqueeze(0).to(device))
            scores = from_device(output[0]["scores"])
            keypoints = from_device(output[0]["keypoints"])
            normalized_keypoints = self._normalize_keypoints(keypoints, scores)
            return normalized_keypoints.astype(np.float32).flatten()