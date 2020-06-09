# Replicating Face Mask Detector
Original work: https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

## The Model
![The Model](https://github.com/Jaldekoa/Replicating-Face-Mask-Detector/blob/master/Face%20Mask%20Detection%20Model.png)

I use MobileNetV2 pretrained model without trainable parameters. Then, I concatenate this model with serveral custom layers ( AveragePooling 2D, Flatten, Dense with 128 relu neurons an 0.5 dropout and last 2 dense relu neurons. 

## The Result
<p align="center">
  <img src="Face Mask Gif.gif"alt="Result"/>
</p>
