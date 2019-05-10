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





