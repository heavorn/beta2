# Ultralytics YOLO 🚀, AGPL-3.0 license

from engine.predictor import BasePredictor
from engine.results import Results
from utils import ops
import torch


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def scale_boxes(self, img_shape, boxes, ori_img_shape):
        # import pdb
        # pdb.set_trace()

        # Calculate the scaling factors
        scaling_factor_width = ori_img_shape[1] / img_shape[1]
        scaling_factor_height = ori_img_shape[0] / img_shape[0]

        # Define predicted bounding boxes for the resized image (x, y, width, height)
        # For this example, we'll assume you have a list of bounding boxes for the resized image.
        # You should replace this with your own predicted bounding box data.

        # Rescale the predicted bounding boxes to the original image size
        predicted_bounding_boxes_original = []
        for box in boxes:
            x, y, width, height = box
            rescaled_x = int(x * scaling_factor_width)
            rescaled_y = int(y * scaling_factor_height)
            rescaled_width = int(width * scaling_factor_width)
            rescaled_height = int(height * scaling_factor_height)
            predicted_bounding_boxes_original.append((rescaled_x, rescaled_y, rescaled_width, rescaled_height))

        return torch.tensor(predicted_bounding_boxes_original, dtype=torch.float32, device=self.device)

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # import pdb
        # pdb.set_trace()
        # img = ops.convert_torch2numpy_batch(img)    

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            # pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            pred[:, :4] = self.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            # results.append(Results(img[i], path=img_path, names=self.model.names, boxes=pred))
        return results
 

