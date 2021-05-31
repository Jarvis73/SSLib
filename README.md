# SSLib
Semantic Segmentation Library (UNet, DeepLabV3)


## Performance

### 1. DeepLabV3

| Dataset  |     Device     | batch size |   lr  | epochs |  mIoU  |
|:--------:|:--------------:|:----------:|:-----:|:------:|:------:|
|CityScapes| Tesla V100 x 1 |     4      | 0.003 |  200   | 77.35  |

| OS = 16 | OS = 8 |  MS  | Flip |  mIoU |
|:-------:|:------:|:----:|:----:|:-----:|
|    ✓   |        |      |       | 77.35 |
|        |    ✓   |      |      |        |
|        |    ✓   |   ✓  |      |        |
|        |    ✓   |   ✓  |   ✓  |       |
