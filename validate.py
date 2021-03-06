import os

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt

from train import get_balloon_dicts, MyDatasetMapper

if __name__ == '__main__':
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("images/"))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.MODEL.DEVICE = 'cpu'

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("balloon_val",)
    predictor = DefaultPredictor(cfg)

    # evaluator = COCOEvaluator("balloon_val", cfg, False, output_dir="./output/")
    # val_loader = build_detection_test_loader(cfg, "balloon_val", MyDatasetMapper(cfg, False))
    # inference_on_dataset(predictor.model, val_loader, evaluator)

    dataset_dicts = get_balloon_dicts("images")

    balloon_metadata = MetadataCatalog.get("balloon_train")

    for d in dataset_dicts:
        f, axarr = plt.subplots(1, 2)
        im = cv2.imread(d["file_name"])[:312, :312, :]
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        preds = outputs["instances"].to("cpu")
        preds.remove('pred_classes')
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        axarr[0].imshow(v.get_image()[:, :, ::-1])

        v = Visualizer(im[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        v = v.draw_dataset_dict(d)
        axarr[1].imshow(v.get_image()[:, :, ::-1])

        plt.show()
