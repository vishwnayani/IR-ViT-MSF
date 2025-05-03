survival_info_df = pd.read_csv('survival_info.csv directory') 
name_mapping_df = pd.read_csv('name_mapping.csv directory')
name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True)
df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")

paths = []
for index, row  in df.iterrows():

    id_ = row['Brats20ID']
    phase = id_.split("_")[-2]

    if phase == 'Training':
        path = os.path.join(config.train_root_dir, id_)
    else:
        path = os.path.join(config.test_root_dir, id_)
    paths.append(path)

df['path'] = paths

train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)
train_data = train_data.loc[train_data['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True, ) 


skf = StratifiedKFold(n_splits=7, random_state=config.seed, shuffle=True) 
for i, (train_index, val_index) in enumerate(skf.split(train_data, train_data["Age"]//10*10)):
        train_data.loc[val_index, "fold"] = i

train_df = train_data.loc[train_data['fold'] != 0].reset_index(drop=True)
val_df = train_data.loc[train_data['fold'] == 0].reset_index(drop=True)
train_data.to_csv("train_data.csv", index=False)

def get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth):
                x1 = (height - crop_height) // 2
                x2 = x1 + crop_height
                y1 = (width - crop_width) // 2
                y2 = y1 + crop_width
                z1 = (depth - crop_depth) // 2
                z2 = z1 + crop_depth
                return x1, y1, z1, x2, y2, z2

def center_crop(data:np.ndarray, crop_height, crop_width, crop_depth):
    height, width, depth = data.shape[:3]
    if height < crop_height or width < crop_width or depth < crop_depth:
        raise ValueError
    x1, y1, z1, x2, y2, z2 = get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth)
    data = data[x1:x2, y1:y2, z1:z2]
    return data

def load_img1(file_path):
    data = nib.load(file_path)
    return data
tumor_core_total=0
peritumoral_edema_total=0
enhancing_tumor_total=0
num_zeros_total=0
for idx in train_data['Brats20ID']:
    root_path = train_data.loc[train_data['Brats20ID'] == idx]['path'].values[0] 
    img_path = os.path.join(root_path +'/' + idx+  '_seg.nii')
    img = load_img1(img_path)
    a = np.array(img.dataobj)
    get_center_crop_coords(240,240,155, 128,128,128)
    a=center_crop(a, 128,128,128)
    b=a.flatten()

    tumor_core=np.count_nonzero(b == 1)
    tumor_core_total=tumor_core_total+tumor_core

    peritumoral_edema=np.count_nonzero(b==2)
    peritumoral_edema_total=peritumoral_edema_total+peritumoral_edema

    enhancing_tumor=np.count_nonzero(b==4)
    enhancing_tumor_total=enhancing_tumor_total+enhancing_tumor

    num_zeros = (b == 0).sum()
    num_zeros_total=num_zeros_total+num_zeros
