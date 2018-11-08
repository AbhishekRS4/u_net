# UNet implementation on Cityscapes dataset

## Notes
* Implementation of UNet with VGG-16 upto pool4
* The image dimension used to train the model is 1024x512
* 15 custom classes used

## Main idea
* Use features from the corresponding encoder stage in decoder [skip connection] by concatenating them. 

## To do
- [x] UNet
- [ ] Compute metrics
- [ ] Sample output

## Reference
* [VGG](https://arxiv.org/abs/1409.1556)
* [U\_Net](https://arxiv.org/pdf/1505.04597)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
