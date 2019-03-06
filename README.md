# CareMentorAI

## Dataset
The MRI dataset of intervertebral discs of spine has been explored in
[`EDA.ipynb`](https://github.com/vessemer/CareMentorAI/blob/master/IPython/EDA.ipynb). 
It's consisted of 891 `.jpg` files with 354 images 'approved for annotation', which in turn represents 
341 images of cervical spine and 13 images of thoracic spine. To format raw `CSV` files with `XML` annotation data 
[`parse_descr.py`](https://github.com/vessemer/CareMentorAI/blob/master/src/scripts/parse_descr.py) has been provided.
Despite there's no obvious clue in dataset description on the images relations, each of them belongs to some 3D MRI cases.
In order to provide a valid estimation of further algorithm, the dataset should be clusterised on such cases. 
To do so the opencv implementation of G Farnebäck optical flow has been employed prior to the weighted trees groing, 
this approach is provided in [`PatientsClustering.ipynb`](https://github.com/vessemer/CareMentorAI/blob/master/IPython/PatientsClustering.ipynb).
Following this we've end up with case clustered MRI images:
![Patints Clustering](https://habrastorage.org/webt/co/qn/be/coqnbezd1edffxtneac2tcuubqm.png)

## Method
To solve the detection task over 2D images one-stage object detector called RetinaNet with ResNet34 as an encoder 
has been used along with focal loss provided by [Tsung-Yi Lin et al.](https://arxiv.org/pdf/1708.02002.pdf). 
The PyTorch implementation is adopted from [yhenon](https://github.com/yhenon/pytorch-retinanet) project. 
Since MRI images are in grayscale we can't use weights of RetinNet pretrained on ImageNet as is, 
to overcome this, weights of the first layer has been summed along the input_chunnel axis (`3x64x7x7 -> 1x64x7x7`).
Due to the relatively small size of the dataset, data augmentation technique was 
[applied](https://github.com/vessemer/CareMentorAI/blob/master/src/modules/augmentations.py):  
![Augmentations](https://habrastorage.org/webt/q3/fu/fh/q3fufhitjmvw245h7jppkzglulo.png)
With that 5-folds cross validation was performed, to estimate the algorithm mAP (mean Average Precision) 
for Object Detection has been tracked:  
![Learning](https://habrastorage.org/webt/_b/vk/39/_bvk390cbyff8nwo1y-y6aczpsg.png)  
 The left image depicts an annotation while the right one — predicted bounding boxes.
## Implementation
The training script is located in [`train.py`](https://github.com/vessemer/CareMentorAI/blob/master/src/scripts/train.py) 
and culd be run as follow:
```
(cxr) #@user:~/CareMentorAI$ python -m src.scripts.train.py
Overlapped keys: 180
Poped keys: []
Summed over: weight
100%|████████████████████████████████████████████████████████████████████| 19/19 [00:21<00:00,  1.13s/it]
100%|██████████████████████████████████████████████████████████████████| 292/292 [00:21<00:00, 13.83it/s]
100%|████████████████████████████████████████████████████████████████████| 62/62 [00:04<00:00, 14.72it/s]
Saved in data/models/retinanet18/fold_0_checkpoint.epoch_128
```

To easier track the learning process [`TensorBoardX`](https://github.com/lanpa/tensorboardX) has been used:  
![TBX scalars](https://habrastorage.org/webt/gd/3l/h0/gd3lh0qwlqrh8tk8hib3_g8ozsk.png)
![TBX images](https://habrastorage.org/webt/jq/o_/8d/jqo_8d92r_scuavvfsnblx-p_hg.png)
