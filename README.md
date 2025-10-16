# Efficient 3D Human Pose Estimation for Edge AI

<p align="center"><img src="figure/htnet.png" width="90%" alt="" /></p>

> [**Efficient 3D Human Pose Estimation for Edge AI**](https://arxiv.org/pdf/2302.),            
> Jialun Cai, Mengyuan Liu, Hong Liu, Wenhao Li, Shuheng Zhou      
> *In IEEE Transactions on Image Processing (TIP), 2025*

<p align="center"><img src="figure/motivation_com.png" width="60%" alt="" /></p>


Here, we are primarily open-sourcing the model framework code for NanoHTNet. The code for the POSECLR contrastive learning paradigm will be introduced and open-sourced in our future work.

## Quick start
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch, test our pretrained models. 


### Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python 3.7+ 
- PyTorch >= 1.10.0


### Dataset setup
Please download the dataset [here](https://drive.google.com/drive/folders/1gNs5PrcaZ6gar7IiNZPNh39T7y6aPY3g) and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

### Evaluating our pre-trained models
Let's take a receptive field of 243 frames and an actual input of 9 frames as an example, the pretrained model is [here](https://drive.google.com/drive/folders/1GABgYBFHbUBOMh_JRdxEtjbTiFxsqGok), please download it and put it in the './ckpt' directory. To achieve the performance in the paper, run:
```
python main.py \
    --reload \
    --previous_dir ./ckpt/demo \
    --frames 243 \
    --gpu 0 \
    --keep_frames 9 
```

### Training your models
If you want to train your own model, run:
```
python main.py --train --frames 243 -n "your_model_name" --keep_frames 9 --gpu 0
```


## Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [MGCN](https://github.com/ZhimingZo/Modulated-GCN)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)
- [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch)
- [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
## Licence

This project is licensed under the terms of the MIT license.
