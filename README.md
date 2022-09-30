# ReCoRoGAN

### ACM MULTIMEDIA, 2022, [ReCoRo: Region-Controllable Robust Light Enhancement by User-Specified Imprecise Masks](https://arxiv.org/)  
ReCoRo is a low-light enhancement approach which allows users to directly specify “where" and "how much" they want to enhance an input low-light image. It also possess resilience to various roughly-supplied user masks.  

Enhancements with both user-specified imprecise and fine matting masks are shown bellow (columns: Mask, Input, ReCoRo(ours), EnlightenGAN, ZeroDCE, DRBN, LIME)
![representive_results](/assets/masks_zoom.png)

### Overal Architecture
![architecture](/assets/architecture.png)

## Environment Preparing
The the code should work on any python >= 3.6 version. 
```pip install -r requirement.txt``` </br>
```mkdir model``` </br>
Download VGG pretrained model from [[Google Drive 1]](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing), and put it into the `model` directory.

### Training

```python train.py -cn recoro_train```

### Testing

```python test.py -cn recoro_train```


If you find this work useful for you, please cite
```
@article{recoro,
  title={ReCoRo: Region-Controllable Robust Light Enhancement by User-Specified Imprecise Masks},
  author={},
  journal={},
  volume={},
  pages={},
  year={},
  publisher={}
}