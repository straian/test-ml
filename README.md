# Computer vision project for removing people from the environment

The project is composed of two parts (two neural nets):

Part 1:

Image segmentation

Identifying people in photos -- trained on 68k photos with humans (Coco dataset http://cocodataset.org/#home) and 5k photos without humans. The photos are RGB and rescaled to 224 x 224.

A encoder - decoder network was used:

2x 2D Conv layers of 224x224x64 + ReLU + Batch Normalization + Max Pooling 2D
2x 2D Conv layers 112x112x128 + ReLU + BN + MP 2D
2x 2D Conv layers 56x56x256 + ReLU + BN + MP 2D
2x 2D Conv layers 28x28x512 + ReLU + BN + MP 2D
1x 2D Conv layer 14x14x512 + ReLU + BN + MP 2D
1x 2D Transpose Conv layer 14x14x512 + ReLU + BN + Upsampling 2D
1x 2D Transpose Conv layer 14x14x512 + ReLU + BN + US 2D
2x 2D Transpose Conv layers 28x28x512 + ReLU + BN + US 2D
2x 2D Transpose Conv layers 56x56x256 + ReLU + BN + US 2D
2x 2D Transpose Conv layers 112x112x128 + ReLU + BN + US 2D
2x 2D Transpose Conv layers 224x224x64 + ReLU + BN + US 2D
1x 2D Transpose Conv layer 2224x224x1 -> Final bit mask indicating 0 or 1 whether the bit is part of a person

Total is about 30M free params


Ran for 30 epochs, overfitting started after ~15 epochs. The photo below shows one pic from the dataset and one from the test set.
![](plot-30.png)



