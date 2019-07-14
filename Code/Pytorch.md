## Function List

`torch.Tensor.contiguous()`:

`torch.Tensor.view()`: 



`nn.Sequential()`: [doc](https://pytorch.org/docs/stable/nn.html#sequential)

`torch.cat`: [doc](<https://pytorch.org/docs/stable/torch.html#torch.cat>)

`torch.squeeze()`: [doc](<https://pytorch.org/docs/stable/torch.html#torch.squeeze>)

`nn.ReLU()`: [doc](<https://pytorch.org/docs/stable/nn.html#relu>)



`nn.AvgPool2d()`: [doc](<https://pytorch.org/docs/stable/nn.html#avgpool2d>)

`nn.BatchNorm2d()`: [doc](<https://pytorch.org/docs/stable/nn.html#batchnorm2d>)





`nn.Conv3d`: [doc](<https://pytorch.org/docs/stable/nn.html#conv3d>)

`nn.ConvTranspose3d()`: [doc](<https://pytorch.org/docs/stable/nn.html#convtranspose3d>)

`nn.BatchNorm3d`: [doc](<https://pytorch.org/docs/stable/nn.html#batchnorm3d>)



`torch.device()`: [doc](<https://pytorch.org/docs/stable/tensor_attributes.html#torch-device>)

`nn.DataParallel()`: [doc](<https://pytorch.org/docs/stable/nn.html#dataparallel>)



`torch.load()`: 

## Gist

**Initialize hyper-parameters**

```python
for m in self.modules():
	if isinstance(m, nn.Conv2d):
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
	elif isinstance(m, nn.Conv3d):
		n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
	elif isinstance(m, nn.BatchNorm2d):
		m.weight.data.fill_(1)
		m.bias.data.zero_()
	elif isinstance(m, nn.BatchNorm3d):
		m.weight.data.fill_(1)
		m.bias.data.zero_()
	elif isinstance(m, nn.Linear):
		m.bias.data.zero_()
```

**Data Paralleling and implement to CUDA**

```python
model = nn.DataParallel(model, device_ids=[0])
model.cuda()
```

**Load existed model**

```python
if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
```

**Print parameter scale**

```python
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
```

**Using pretrained model to evaluate Images**

```python
from __future__ import print_function, division

from PIL import Image, ImageDraw
import torch
import torch.nn as nn

import torchvision

import torchvision.transforms as T


W = 800
H = 800

def get_transform():
    transforms = []
    transforms.append(T.Resize((W,H),2))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


# Load data
img = Image.open("data/two-persons.jpg")

# Load pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_nums = torch.cuda.device_count()
if device_nums > 1:
    model = nn.DataParallel(model, device_ids=range(device_nums))
model = model.to(device)

# Evaluation
model.eval()
x = get_transform()(img)
y = model(x.view(-1,3,W,H))

print(y)

# Visualization
w, h = img.size
w_ratio = w / W
h_ratio = h / W
draw = ImageDraw.Draw(img)
for idx, (x1, y1, x2, y2) in enumerate(y[0]['boxes'].detach().numpy()):
    if y[0]['scores'].detach().numpy()[idx] > 0.9:
        draw.rectangle(((x1 * w_ratio, y1 * h_ratio), (x2 * w_ratio, y2 * h_ratio)), fill=None, width=2)
img.show()
```





