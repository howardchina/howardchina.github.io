---
layout: post
title:  SPM Anatomical Parcellation
date:   2019-01-04 15:27:00 +0800
categories: [brain,tools]
---

N. Tzourio-Marozyer et. al 2002

## Automated Anatomical Labeling of Activations in SPM Using a Macroscopic Anatomical Parcellation of the MNI MRI Single-Subject Brain

**Abstract**

Anatomical parcellation of T1 volume provided by the Montreal Neurological Institute  (MNI) was performed. ROI were then drawn manually with the same software every 2 mm on the axial slices of the high-resolution MNI single subject. The 90 anatomocal volumes of interest (AVOI), where 45 AVOI in each hemisphere, were reconstructed and assigned a label. Three procedures to perform the automated anatomical labeling of functional studies are proposed: (1) labeling of an extremum defined by a set of coordinates, (2) percentage of voxels belonging to each of the AVOI intersected by a sphere  centered by a set of coordinates, and (3) percentage of voxels belonging to each of the AVOI intersected by an activated cluster.

**INTRODUCTION**

one of  major goals: establish relationships between brain structures and their functions; reduce anatomical and functional variability bewteen subjects.

spatial registration and normalization

**SPM99:** spatial normalization; average of 152 brains;

* lack of detailed anatomical features
* not anatomically labeled

**Talairach atlas** (Talairach and Tournoux, 1998): report localization of activations detected in functional imaging studies.

* detailed anatomical description, includeing **Brodmann's areas** (**BA**)
  * ![brodmann area]({{site.url}}/static/img/posts/320px-Gray726-Brodman.png "outer")
  * ![brodmann area]({{site.url}}/static/img/posts/320px-Gray727-Brodman.png "inner")
* inaccurate
  * coarse (4mm)
  * a single hemisphere is labeled
  * lots of ambiguities when the point falls in between different brain areas

Automatic labeling of activations based on the Talairach atlas (**Lancaster et al., 2000**)

* **hierarchical classification including BA**
* **offers a reference frame for activation labeling**
* inaccurate Figs. 1A, 1B and 1C.

This study:

* automated anatomical labeling of activations detected with PET or fMRI based on an anatomical parcellation of the MNI single-subject brain (Tzourio et al., 1997)
* the purpose of this work was to suppress the confusion existing in the literature regarding the relationship between a set of coordinates and its anatomical label
* 

![1546413486843]({{site.url}}/static/img/posts/1546413486843.png)  

Fig. 1. Talairach atlas is inaccurate.

![1546413521319]({{site.url}}/static/img/posts/1546413521319.png)

Fig. 2. Activation atlas proposed.

**OVERVIEW OF THE METHOD**

*MNI Single-Subject Images*

MRI brain template from Montreal Neurological Institue database.
* a young man whose brain was scanned 27 times using T1
* corrected[Sled, 1998] and spatially normalized[Collins, 1994]
* the average of the 27 acquisitions are used with MNI web brain simulator (Collions, 1998) and SPM99
* segmentation in eight classes including gray matter, white matter, cerebrospinal fluid, fat, muscle/skin, skin, skull, and glial matter.
* "single_subj_T1" in the SPM package

*Sulci Delineation*

* sulci courses were drawn on the 3D rendering of the cortical surfaces (Fig. 2).
* software (Voxeline, Diallo, 1998)
* 3D tracking and drawing of anatomical landmarks both on the external surface of the hemisphere and on any incidence.

***Regions Drawing***

* draw manually every 2 mm on axial slices (Fig. 3 and 4).
* sulci landmarks were used as the limits of the ROI on the outer surface of the brain.
* inner boundaries were traced along the sulci internal course, starting from the landmark.

![1546413558671]({{site.url}}/static/img/posts/1546413558671.png)

![1546413670001]({{site.url}}/static/img/posts/1546413670001.png)

*3D Anatomical Volumes of Interest (AVOI)*

* each region was two dimensionally filled using 4邻域联通
* each anatomical regiona 3D AVOI was created
* the 1-mm<sup>3</sup> volume was calculated using the nearest neighbor interpolation
* Each AVOI was then given a gray-level code

*Automated Anatomical Labeling*

three procedures for automatic anatomical labeling of the functional activation map:

* local maxima labeling
* extended local maxima labeling
* cluster labeling

preparation: spatially normalized using MNI template

1. functional map is thresholded
2. both the local maxima and the clusters are extracted

illustrated in Fig. 5.

![1546413958522]({{site.url}}/static/img/posts/1546413958522.png)

*SPM Anatomical Labeling Interface*

**MNI SINGLE-SUBJECT BRAIN PARCELLATION**

**Sulci Identification**

![1546586246315]({{site.url}}/static/img/posts/1546586246315.png)

*Lateral Surface of the Hemispheres*

* the central region
* the frontal lobe
* the temporal lobe
* the parietal lobe and occipital lobe

*Internal Surface of the Hemispheres*

* Frontal lobe
* Occipital lobe

*Inferior Surface of the Hemisphere*

* Frontal lobe
* Temporal lobe

**Regions Definition**

*Central Region*

* Precentral gyrus and postcentral gyrus
* Rolandic operculum

*Frontal Lobe Lateral Cortex*

* The superior, middle, and inferior frontal gyri.
* The medial surface of the frontal lobe
* The orbital part of the frontal lobe

*Temporal Lobe*

* Planimetry
* Lateral cortex

Parietal Lobe

* Lateral surface
* Medial surface

Occipital Lobe

* Lateral surface
* Medial surface

The Limbic Lobe

* The cingulate regions
* The temporal pole
* The hippocampal and parahippocampal regions

*The Insula and Subcortical Gray Nuclei*

**Application to Functional Activation Labeling**

![1546586215513]({{site.url}}/static/img/posts/1546586215513.png)

**DISSCUSSION**

*Anatomical Issures*

* MNI single-subject brain
* AVOI definition
* Comparison with other parcellations

*Labelling of Functinal Areas*

* SPM interface
