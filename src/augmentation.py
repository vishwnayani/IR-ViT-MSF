from monai.transforms import (
    Compose,
    RandAffined,
    RandGaussianNoised,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandFlipd,
    RandRotate90d,
    ToTensord
)

# Define MONAI transforms for 3D data
train_transforms = Compose([
    #EnsureChannelFirstd(keys=['image', 'mask']),  # Ensure data has channel dimension
    ScaleIntensityd(keys=['image']),  # Normalize intensity values
    RandAffined(
        keys=['image', 'mask'],
        mode=('bilinear', 'nearest'),  # Use bilinear for image, nearest for mask
        prob=0.5,
        spatial_size=(128, 128, 128),  # Output size
        rotate_range=(0, 0, np.pi/15),  # Random rotation
        scale_range=(0.1, 0.1, 0.1)  # Random scaling
    ),
    RandGaussianNoised(keys=['image'], prob=0.5, std=0.01),  # Add Gaussian noise
    RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),  # Random flip
    RandRotate90d(keys=['image', 'mask'], prob=0.5, max_k=3),  # Random 90-degree rotation
    ToTensord(keys=['image', 'mask'])  # Convert to PyTorch tensors
])

class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str="test", transform=None):
        self.df = df
        self.phase = phase
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        if transform is None:
            self.transform = train_transforms if phase == "train" else Compose([])
        else:
            self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        
        # Load and preprocess image volumes
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)
            img = self.center_crop(img, 128, 128, 128)
            img = self.normalize(img)
            images.append(img)
            
        # Stack modalities to create 4D tensor (4, 128, 128, 128)
        img = np.stack(images)  # Shape: (4, 128, 128, 128) = (C, H, W, D)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        img=img.astype(np.float32)

        if self.phase != "test":
            # Load and preprocess mask
            mask_path = os.path.join(root_path, id_ + "_seg.nii")
            mask = self.load_img(mask_path)
            mask = self.center_crop(mask, 128, 128, 128)
            mask = self.preprocess_mask_labels(mask)  # Shape: (3, 128, 128, 128)
            mask = mask.astype(np.float32)
            aug_img, aug_mask = img, mask 

            # Apply MONAI transforms
            
            data = {'image': img, 'mask': mask}
            transformed = self.transform(data)
            aug_img = transformed['image']
            aug_mask = transformed['mask']

            return {
                "Id": id_,
                "image": img,
                "mask": mask,
                "aug_image": aug_img,
                "aug_mask": aug_mask,
            }

        return {
            "Id": id_,
            "image": torch.from_numpy(img),
        }

    # Keep other helper methods the same
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def get_center_crop_coords(self, height, width, depth, crop_height, crop_width, crop_depth):
        x1 = (height - crop_height) // 2
        x2 = x1 + crop_height
        y1 = (width - crop_width) // 2
        y2 = y1 + crop_width
        z1 = (depth - crop_depth) // 2
        z2 = z1 + crop_depth
        return x1, y1, z1, x2, y2, z2

    def center_crop(self, data: np.ndarray, crop_height, crop_width, crop_depth):
        height, width, depth = data.shape[:3]
        if height < crop_height or width < crop_width or depth < crop_depth:
            raise ValueError
        x1, y1, z1, x2, y2, z2 = self.get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth)
        data = data[x1:x2, y1:y2, z1:z2]
        return data
        
    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1  # Label 1 = necrotic / non-enhancing tumor core
        mask_WT[mask_WT == 2] = 1  # Label 2 = peritumoral edema
        mask_WT[mask_WT == 4] = 1  # 

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))  # Reorder axes for visualization
        return mask

