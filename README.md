# UNet implementation on Cityscapes dataset

## Notes
* Implementation of UNet with VGG-16 upto pool4
* The image dimension used to train the model is 1024x512
* 15 custom classes used

## Main idea
* Use features from the corresponding encoder stage in decoder [skip connection] by concatenating them.

## Intructions to run
* To run training use
```
python3 src/u_net_train.py --help
```
* To run inference use
```
python3 src/u_net_infer.py --help
```

## Visualization of results
* [UNet](https://youtu.be/s6FitFbnxXQ)

## Reference
* [VGG](https://arxiv.org/abs/1409.1556)
* [UNet](https://arxiv.org/pdf/1505.04597)
* [UNet Project](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
