# ReCoRoGAN

### ACM MULTIMEDIA, 2022, ReCoRo: Region-Controllable Robust Light Enhancement by User-Specified Imprecise Masks

[[Paper PDF]](https://github.com/Picsart-AI-Research/ReCoRo-Controllable-Low-Light-Image-Enhancement/blob/master/recoro_paper.pdf)

ReCoRo is a low-light enhancement approach which allows users to directly specify “where" and "how much" they want to enhance an input low-light image. It also possess resilience to various roughly-supplied user masks.  

Enhancements with both user-specified imprecise and fine matting masks are shown bellow (columns: Mask, Input, ReCoRo(ours), EnlightenGAN, ZeroDCE, DRBN, LIME)
![representive_results](/assets/masks_zoom.png)

### Overal Architecture
![architecture](/assets/architecture.png)

## Environment Preparing
The code should work on any python >= 3.6 version. 
```pip install -r requirement.txt``` </br>
```mkdir model``` </br>
Download VGG pretrained model from [[Google Drive 1]](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing), and put it into the `model` directory.

### Training

```python train.py -cn recoro_train```

### Testing

```python test.py -cn recoro_train```


If you find this work useful for you, please cite

```
@inproceedings{xu2022recoro,
  title={ReCoRo: Re gion-Co ntrollable Ro bust Light Enhancement with User-Specified Imprecise Masks},
  author={Xu, Dejia and Poghosyan, Hayk and Navasardyan, Shant and Jiang, Yifan and Shi, Humphrey and Wang, Zhangyang},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1376--1386},
  year={2022}
}
```
