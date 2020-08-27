# Detectron2

## credits
Credits are given to the [detectron2 tutorials](https://detectron2.readthedocs.io/tutorials/getting_started.html).


## environment
Based on PyTorch 1.6, CUDA 10.1.
Detectron2 source code is downloaded from [detectron2](https://github.com/facebookresearch/detectron2). 


## datasets

### builtin datasets
To use builtin datasets, set the environment variable `DETECTRON2_DATASETS` by `export DETECTRON2_DATASETS=/home/david/datasets`. To use COCO detection datasets, the structure of the path should be like this:
```
- DETECTRON2_DATASETS/
    - coco/
        - annotations/
            - instances_train2017.json
            - instances_val2017.json
        - train2017/
            - train images
        - val2017/
            - val images
```

## build a model

### build a model from cfg
This will build a model from cfg and fill it with random parameters.
```
from detectron2.modeling import build_model
model = build_model(cfg)
```

### model input format
The input should be list[dict] format. Every dict corresponds to an image. The default `DatasetMapper` will output a dict having the following format. The dataloader will use list() to collate the dict. The dict should contain the keys:
- image: tensor in (C,H,W) format. Normalization will be performed in model if there is any.
- height: **desired** output height
- width: **desired** output width
- instances: an **detectron2.structures.Instances** object, containing the following fields:
    - gt_boxes: a **detectron2.structure.Boxes** object (N*4 Tensor)
    - gt_classes: Tensor of long type, N labels
    - gt_masks: a **detectron2.structure.PolygonMasks** or **detectron2.structure.BitMasks** object.
    - gt_keypoints: a **detectron2.structure.Keypoints** object.


### model output format
When the model is in training mode, the builtin models output dict[str -> ScalarTensor] with all the losses.
When the model is in inference mode, the builtin models output list[dict], one dict for one image with the following keys:
- instances: Instances object with the follow fields:
    - pred_boxes: a **detectron2.structure.Boxes** object
    - scores: Tensor, size = N
    - pred_classes: Tensor, N labels
    - pred_masks: Tensor, size = (N, H, W)
    - pred_keypoints: Tensor (N, num_keypoint, 3)


### call a model
If the model is in training mode, the model should be under an `EventStorage`. The training statistics will be stored there.
```
from detectron2.utils.events import EventStorage
with EventStorage() as storage:
    losses = model(inputs)
```

If the model is in evaluation mode, you can use `DefaultPredictor` or directly the model.
- use the model directly:
```
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

### access the `EventStorage`
During training, inside the model, call `get_event_storage` to get the storage.
```
from detectron2.utils.events import get_event_storage

if self.training:
    value = ... # compute the value to log
    storage = get_event_storage()
    storage.put_scalar("metric_name", value)
```

### write a new component
Two steps:
1. write the component
2. register the component, so that the cfg can use it
For example, to write a customized backbone:
```
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class ToyBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        ...
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7)
    
    def forward(self, image):
        return {"conv1": self.conv1(image)}
    
    def output_shape(self):
        return {"conv1": ShapeSpec(channels=64, stride=16)}

# use it like this
cfg = ... # get the cfg
cfg.MODEL.BACKBONE.NAME = 'ToyBackbone'
model = build_model(cfg)
```


## checkpoints

### load a checkpoint file
This will load a checkpoint, usually from `cfg.MODEL.WEIGHTS`
```
from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(file_path_or_url)
```

### save a checkpoint
This will save a checkpoint to `output/model_123`
```
from detectron2.checkpoint import DetectionCheckpointer
checkpointer = DetectionCheckpointer(model, save_dir='output')
checkpointer.save('model_123')
```


## train a model

> Before training, the datasets folder needs to be set in the environment, e.g. `export DETECTRON2_DATASETS=/home/david/datasets`.

### command line

1. go to tools/
2. run `./train_net.py --num-gpus 1 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025`

In the arguments:
- `--num-gpus` will set the gpu number per machine.
- `--config-file` will set the config file.
- `SOLVER.IMS_PER_BATCH` will set the total image number per batch, usually it's 2 per GPU, so it should be `2*num_gpus`
- `SOLVER.BASE_LR` will set the base learning rate, usually it's `0.0025*num_gpus`


## evaluate a model

### evaluate on command line

1. go to tools/
2. run `./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --eval-only MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl`

In the arguments:
- `--config-file` will set to the config file.
- `--eval-only` will set the model to evaluation mode.
- `MODEL.WEIGHTS` will set the weights to use when evaluating the model.



## inference using a model

### inference from command line

> **OpenCV is needed.**

1. go to demo/
2. run `python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --webcam --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl`

In the arguments:
- `--config-file` is to set the config file path
- `--webcam` is to set the input from webcam, other options are: `--input file1.jpg file2.jpg` or `--video-input video.mp4`
- `--opts` will have pairs of configurations followed
- if inference on cpu, add `MODEL.DEVICE cpu` after `--opts`
- `--config-file` the configs are for training, use `MODEL.WEIGHTS` to use a model with weights from model zoo.
- `--output` can save outputs to a file (for `--input file1.jps`) or to a folder (for `--webcame` or `--video-input`).

### inference from python script
`detectron2.evaluation.inference_on_dataset` will run model on dataloader using evaluator(s). COCOEvaluator can evaluate AP for box detection and so on.


## configs

### get config, update config and merge config
```
from detectron2.config import get_cfg
cfg = get_cfg()
cfg.xxx = yyy
cfg.merge_from_file('abc.yaml')
cfg.merge_from_list(['MODEL.WEIGHTS', 'weights.pth'])

print(cfg.dump())
```

### versioning
use something like `version: 2` in the config file for backward compatibility.

### reusability
use `_BASE_` to reuse some of the configs, instead of copying and pasting them around.