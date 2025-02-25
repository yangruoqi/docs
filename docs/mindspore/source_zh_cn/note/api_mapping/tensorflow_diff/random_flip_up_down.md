# 比较与tf.image.random_flip_up_down的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/random_flip_up_down.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.image.random_flip_up_down

```python
tf.image.random_flip_up_down(
    image,
    seed=None
)
```

更多内容详见[tf.image.random_flip_up_down](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/random_flip_up_down)。

## mindspore.dataset.vision.c_transforms.RandomVerticalFlip

```python
class mindspore.dataset.vision.c_transforms.RandomVerticalFlip(
    prob=0.5
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.RandomVerticalFlip](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomVerticalFlip.html#mindspore.dataset.vision.c_transforms.RandomVerticalFlip)。

## 使用方式

TensorFlow：随机垂直翻转图像，概率为0.5，随机种子可通过入参指定。

MindSpore：随机垂直翻转图像，概率可通过入参指定，随机种子需通过 `mindspore.dataset.config.set_seed` 全局设置。

## 代码示例

```python
# The following implements RandomVerticalFlip with MindSpore.
import numpy as np
import mindspore.dataset as ds

ds.config.set_seed(57)
image = np.random.random((28, 28, 3))
result = ds.vision.c_transforms.RandomVerticalFlip(prob=0.5)(image)
print(result.shape)
# (28, 28, 3)

# The following implements random_flip_up_down with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.random_flip_up_down(image, seed=57)
print(result.shape)
# (28, 28, 3)
```
