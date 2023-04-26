# PyTorch Quantile Tokenization

PyTorch Quantile Tokenization is a module for easy and efficient continuous numbers' quantile tokenization. This module can be used for any 1D or 2D vector, including time-series data. This module is inspired by the tokenization technique used in Uber's [DeepETA](https://www.uber.com/en-FI/blog/deepeta-how-uber-predicts-arrival-times/).

The module works in three simple steps:
1. Define the module:
```python
tokenizer = QuantileTokenization(q_num=q_num, f_num=f_num, embed_dim=embed_dim, mode=mode)
```
2. Fit the feature distribution on the entire training dataset:
```python
tokenizer.fit(tensor)
```
- Tokenize the data:
```python
tokenized_tensor = tokenizer(tensor)
```

The `fit` step calculates the quantiles for the input features and stores it as a torch Parameter. This enables the tokenizer object to be saved and reloaded for future use, without having to recalculate the quantiles each time.

Use cases:
1. Time-series forecasting 
2. Time-series classification
3. Tabular data classification/regression

# Examples

### Example 1

```python
>>> from modules import QuantileTokenization
>>>
>>> f_num = 2
>>> train_features = torch.rand(10_000, f_num)
>>> tokenizer = QuantileTokenization(q_num=3, f_num=f_num, embed_dim=2, mode='flatten').fit(train_features)
>>> out = tokenizer(train_features)
>>> print(out)
tensor([[-0.0365,  0.1279, -0.2751,  1.3138],
        [ 0.4980,  0.5532, -0.2751,  1.3138],
        [ 0.4269, -1.1461, -0.2134, -0.2625],
        ...,
        [ 0.4269, -1.1461,  1.6891, -1.0809],
        [-0.0365,  0.1279, -0.2751,  1.3138],
        [ 0.4980,  0.5532, -0.2751,  1.3138]], grad_fn=<ReshapeAliasBackward0>)

>>> buckets = tokenizer.bucketize(train_features)
>>> print(buckets)
tensor([[3, 5],
        [2, 5],
        [1, 6],
        ...,
        [1, 4],
        [3, 5],
        [2, 5]])
  
# You can save and load the module state:
>>> torch.save(tokenizer.state_dict(), 'test.torch')
>>>
>>> tokenizer = QuantileTokenization(q_num=3, f_num=f_num, embed_dim=2, mode='flatten')
>>> tokenizer.load_state_dict(torch.load('test.torch'))
>>>
>>> buckets = tokenizer.bucketize(train_features)
>>> print(buckets)
tensor([[3, 5],
        [2, 5],
        [1, 6],
        ...,
        [1, 4],
        [3, 5],
        [2, 5]])
```

### Example 2

```python
>>> from modules import QuantileTokenization
>>>
>>> t = torch.rand(128, 64, 4)
>>> tokenizer = QuantileTokenization(q_num=10, f_num=4, embedding_dim=2, mode='flatten').fit(t.flatten(0, 1))
>>> print(t[0, :2])
tensor([[0.9791, 0.0989, 0.1914, 0.2806],
        [0.4861, 0.6315, 0.1477, 0.5515]])
>>> buckets = tokenizer.bucketize(t)
>>> print(buckets[0, :2])
tensor([[10, 11, 22, 33],
        [ 5, 17, 22, 36]])
>>> print(tokenizer.embed(buckets)[0, :2])
tensor([[-1.2583,  0.7034,  0.2587,  0.9427,  0.9104,  1.3444,  0.2164,  0.1352],
        [ 0.9450,  2.6574,  0.7556, -0.6013,  0.9104,  1.3444,  0.8862,  0.6398]],
    grad_fn=<SliceBackward0>)
```
