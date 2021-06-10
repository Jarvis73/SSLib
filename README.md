# SSLib
Semantic Segmentation Library (UNet, DeepLabV3)


## 1. Results

* CityScapes

```bash
# Train
python main.py train with lr=0.003 epochs=200

# Test: `tta` means test-time augmentation, indicating multi-scale and flip (MS+Flip)
#       `ckpt_id` means the experiment id, linking to checkpoints of the experiment 
python main.py test with tta=True ckpt_id=<ExpID>
```

|        Model      | batch size | train/val OS |  mIoU  |   TTA   |
|:-----------------:|:----------:|:------------:|:------:|:-------:|
|DeepLabV3+ResNet101|     4      |       8      |  77.35 |**78.45**|

* PASCAL VOC

```bash
# Train Stage1: `voc` means applying the configurations corresponding to the PASCAL VOC dataset.
#               Global search `voc()` for details.
python main.py train with voc split=trainaug epochs=45 lr=0.007 output_stride=16 bs=16

# Test Stage1
python main.py test with voc output_stride=16 tta=True ckpt_id=<ExpID_1>

# Train Stage2
python main.py train with voc split=train epochs=45 lr=0.001 output_stride=8 bs=8 freeze_bn=True ckpt_id=<ExpID_1>

# Test Stage2
python main.py test with voc output_stride=8 tta=True ckpt_id=<ExpID_2>
```

|        Model      | batch size | train/val OS | WarmUp |  mIoU  |   TTA   |
|:-----------------:|:----------:|:------------:|:------:|:------:|:-------:|
|DeepLabV3+ResNet101|     16     |      16      |        | 75.43  |  77.28  |
|DeepLabV3+ResNet101|     8      |       8      |    ✓   | 78.37  |**80.42**|

