### DeepLabV3+

<p>
All credit for the original Keras implementation of DeepLabV3+ used in this thesis is due to [github user bonlime](https://github.com/bonlime/keras-deeplab-v3-plus). The license provided in the original repository is included here.
</p>

<p>
**Implemented modifications:**
- Remove all parts not necessary if backbone is not >xception.
- Support for (None, None, 3) input shapes (heavily inspired by previous commits to original model).
</p>
