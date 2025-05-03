# IR-ViT-MSF
Automated 3D Brain Tumor Segmentation
A deep learning framework designed to segment brain tumors from 3D MRI volumes by integrating Inception-Residual blocks, Vision Transformers (ViT), and Multi-Scale Fusion techniques.

---

## 📌 Project Description

This project explores a novel hybrid architecture for medical image segmentation—specifically, segmenting brain tumors from MRI scans. The proposed architecture, IR-ViT-MSF, combines:
- Inception-Residual (IR) blocks for capturing multi-scale contextual features,
- Vision Transformer (ViT) for learning long-range dependencies,
- Multi-Scale Fusion for integrating hierarchical features effectively.

The model was implemented and trained using the BraTS2018 dataset and evaluated on multiple segmentation metrics including Dice Score, IoU, Sensitivity, and Specificity.

---

## 📂 Dataset Description

**Source:** [BraTS 2020 Challenge](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

**External Validation:** [BraTS2018](https://www.kaggle.com/datasets/harshitsinghai/miccai-brats2018-original-dataset)

Note: Subset of BraTS2018 was used which consisted of randomly selected patients IDs for external validation.

### 📁 Expected Dataset Structure

Your dataset directory should follow this structure:
```
Dataset/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS18_Training_002/
│   └── ...
```

### 🧪 Modalities Used
- T1-weighted (T1)
- T1-weighted with contrast (T1ce)
- T2-weighted (T2)
- Fluid-Attenuated Inversion Recovery (FLAIR)


---

## 🧰 Installation
The experiment was run using Kaggle notebook. Similar environment can be set up using:
<pre lang="markdown">  !pip install -r requirements.txt </pre>

---

## Acknowledgements
- BraTS2020
- BraTS2018
- Co-authors of the research paper(under review)





