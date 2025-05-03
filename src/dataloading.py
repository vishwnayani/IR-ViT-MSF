def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
    transform = None,
    
):

    df = pd.read_csv(path_to_csv)

    train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    df = train_df if phase == "train" else val_df
        # Handle dataset splitting

    dataset = dataset(df, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )
      return dataloader


train_dataloader = get_dataloader(BratsDataset, "train_data.csv", phase="train", transform=train_transforms)
val_dataloader = get_dataloader(BratsDataset, "train_data.csv", phase="valid", transform=None, fold = 0)
