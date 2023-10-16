# CompNet: Complementary network for single-channel speech enhancement

Complementary network for single-channel speech enhancement (CompNet)](https://doi.org/10.1016/j.neunet.2023.09.041) using PyTorch.

## ToDo
- [x] Real-time version
- [x] Update trainer

## Usage

Training:

```
python -B main.py \
    --config="configs/train_config.toml" \
```

Inference:

```
python taylorsenet_decode.py 
```



## Dependencies

- Python==3.\*.\*
- torch==1.\*

## References

- [A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf)

  
