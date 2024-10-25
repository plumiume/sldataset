# TL; DR
```python
sldataset # module
inputs # list[Tensor]
glosses # list[list[str]]

standard_scaler = sldataset.standard_scale(inputs)
labels, label_encoder = sldataset.label_encode(glosses)

indices # list[int]

```

## 1. prepare data and gloss

