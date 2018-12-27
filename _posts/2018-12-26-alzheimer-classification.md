---
layout: post
title: Alzhemier's Analysis with Deep Learning
date:   2018-12-26 20:43:00 +0800
categories: [brain]
---

Thank you Mr. [Marcia Hon](https://github.com/marciahon29/Ryerson_MRP) for your contribution. The original paper, **"Towards Alzheimer;s Disease Classification through Transfer Learning"**, was previously posted with several brain neural related papers  at [here]({{site.url}}/brain/2018/12/20/brain-neural.html).

**ref**: M. Hon, N.M. Khan. Towards Alzheimer&#39;s Disease Classification through Transfer Learning. IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2017.

Github code **overview**:

```shell
fchollet
InceptionV4_Experiment_Models
InceptionV4_FINETUNING
InceptionV4_FineTunnign_ALL
Initial_Exploration
MPR_Results.pdf
MRI_32_Images
MRP_Abstract.pdf
MRP_LiteratureReviewAndExploratoryDataAnalysis.pdf
MRP_ProjectReport.docx
PythonGUI
README
VGG16_Experiment
VGG16_Experiment_Models
VGG16_None_Experiments
VGG19_Experiment_Models
```

interpret files one by one.

---

*MRP_Abstract.pdf*

classify and predict AD based on MRI and PET images of brain.

---

*MPR_Results.pdf*

**Data Preparation**

Alzheimer's was determined by the "CRD (Clinical Dementia Rating)" variable:

* 0 = nondemented
* 0.5 = very mild dementia
* 1 = mild dementia
* 2 = moderate dementia

6400 images in total, thus meaning 3200 for Alzheimer's and 3200 for non-Alzheimer's.

5-fold. 4 buckets together for train has 2560 images for each class and 1 bucket of 640 images for test images for each class. Files saved in https://github.com/marciahon29/Ryerson_MRP/tree/master/MRI_32_Images. Format "YAL" = Yes Alzheimers and "NAL" = No Alzheimers.

```
data_##:
	train
		alzheimers
			YAL0001.jpg
			YAL0002.jpg
			etc...
		nonalzheimers
			NAL0001.jpg
			NAL0002.jpg
			etc...
	validation
		alzheimers
			YAL0001.jpg
			YAL0002.jpg
			etc...
		nonalzheimers
			NAL0001.jpg
			NAL0002.jpg
			etc...
```

**Environment Preparation**

Ubuntu 16, CPU (limitation in hardward)

The following are the steps for getting Python (2.7), Keras, and Tensorflow to work in Ubuntu:

```shell
1. python –v (make sure it is 2.7)
2. sudo apt install python-pip
3. pip install numpy
4. pip install pillow
5. pip install scipy
6. pip install keras
7. pip install h5py
8. sudo apt-get install python-pip python-def
9. pip install tensorflow
10. suto apt-get install python-opencv
```

**Algorithm**

Transfer learning: backbone VGG16

code: https://github.com/marciahon29/Ryerson_MRP/tree/master/VGG16_Experiment_Models/Scripts

```
img_width, img_height = 150, 150
train_data_dir = 'data_00/train'
validation_data_dir = 'data_00/validation'
nb_train_samples = 5120
nb_validation_samples = 1280
epochs = 100
batch_size = 40
```

Here is the sample output:

```
Found 5120 images belonging to 2 classes.
Found 1280 images belonging to 2 classes.
Train on 5120 samples, validate on 1280 samples
…
…
…
4760/5120 [==========================>...] - ETA: 0s - loss: 0.1817 - acc: 0.9193
4840/5120 [===========================>..] - ETA: 0s - loss: 0.1813 - acc: 0.9196
4920/5120 [===========================>..] - ETA: 0s - loss: 0.1821 - acc: 0.9199
5000/5120 [============================>.] - ETA: 0s - loss: 0.1813 - acc: 0.9204
5080/5120 [============================>.] - ETA: 0s - loss: 0.1813 - acc: 0.9199
5120/5120 [==============================] - 5s - loss: 0.1808 - acc: 0.9201 - val_loss:
0.1911 - val_acc: 0.9406
```

---



[20] Used Alzheimer’s classification using CNN. They adopted LeNet and GoogleNet which successfully predicted Alzheimer’s. 

[21] Uses state-of-the-art deep learning-based pipelines to distinguish Alzheimer’s in MRI and fMRI using GPU-based high performance computer platforms. 

[22] Details what Alzheimer’s is on a cellular/biological level. This is important in order to understand what the Alzheimer’s MRI images are like. 

[23] Alzheimer’s classification using AlexNet (with TensorFlow) and ADNI is used. Many important components of the MRP will use this article.

**Exploratory Data Analysis**

OASIS. This server consists of two projects:
1. OASIS: Cross-sectional MRI Data in Young, Middle Aged, Nondemented and Demented Older Adults
2. OASIS: Longitudinal MRI Data in Nondemented and Demented Older Adults

To simply imaging data for this project, **only the middle images** for **Axial (Transverse), Sagittal, and Coronal axis** were selected. The location is at:
**OAS1####MR1\PROCESSED\MPRAGE\T88_111***. And these GIF images have the following
naming convention:

```
1. OAS1_####_MR1_mpr_n4_anon_111_t88_gfc_tra_90 (Transverse)
2. OAS1_####_MR1_mpr_n4_anon_111_t88_gfc_sag_95 (Sagittal)
3. OAS1_####_MR1_mpr_n4_anon_111_t88_gfc_cor_110 (Coronal)
```



Each patient, has all three images. The following is an image that visually presents the meaning of the different axis.

![1545835223320]({{site.url}}/static/img/posts/1545835223320.png)

Alzheimer's can be correctly diagnosed via neuroimaging as it is characterized as **atrophy** of certain parts of the brain.

![1545834959618]({{site.url}}/static/img/posts/1545834959618.png)

In the next image, the **left** is of a **healthy** brain and the **right** is one with **Alzheimer's**.

![1545835003779]({{site.url}}/static/img/posts/healthybrainversealzheimers.png)

---

https://github.com/marciahon29/Ryerson_MRP/blob/master/VGG16_Experiment/Scripts/VGG16_bottleneck_data_00.py 

```python
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_00.h5'
train_data_dir = 'data_00/train'
validation_data_dir = 'data_00/validation'
nb_train_samples = 5120
nb_validation_samples = 1280
epochs = 100
batch_size = 40


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()
```

Replace original top layers by new fully connected layers. 