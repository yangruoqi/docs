# Function Differences with torch.nn.BatchNorm1d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BatchNorm1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.BatchNorm1d

```python
class torch.nn.BatchNorm1d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)
```

For more information, see [torch.nn.BatchNorm1d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.BatchNorm1d).

## mindspore.nn.BatchNorm1d

```python
class mindspore.nn.BatchNorm1d(
    num_features,
    eps=1e-05,
    momentum=0.9,
    affine=True,
    gamma_init="ones",
    beta_init="zeros",
    moving_mean_init="zeros",
    moving_var_init="ones",
    use_batch_statistics=None)
)
```

For more information, see [mindspore.nn.BatchNorm1d](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.BatchNorm1d.html#mindspore.nn.BatchNorm1d).

## Differences

PyTorch：The default value of the momentum parameter used for running_mean and running_var calculation is 0.1.

MindSpore：The default value of the momentum parameter is 0.9, and the momentum relationship with Pytorch is 1-momentum, that is, when Pytorch’s momentum value is 0.2, MindSpore’s momemtum should be 0.8. Parameter beta, gamma, moving_mean and moving_variance correspond to Pytorch's bias, weight, running_mean and running_var parameters respectively.

## Code Example

```python
# The following implements BatchNorm1d with MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor

net = nn.BatchNorm1d(num_features=4, momentum=0.8)
x = Tensor(np.array([[0.7, 0.5, 0.5, 0.6],
                     [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = net(x)
print(output)
# Out:
# [[ 0.6999965   0.4999975  0.4999975  0.59999704 ]
#  [ 0.4999975   0.399998   0.59999704 0.89999545 ]]


# The following implements BatchNorm1d with torch.
input_x = torch.tensor(np.array([[0.7, 0.5, 0.5, 0.6],
                                 [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
m = torch.nn.BatchNorm1d(4, momentum=0.2)
output = m(input_x)
print(output)
# Out:
# tensor([[ 0.9995,  0.9980, -0.9980, -0.9998],
#         [-0.9995, -0.9980,  0.9980,  0.9998]],
#        grad_fn=<NativeBatchNormBackward>)
```
