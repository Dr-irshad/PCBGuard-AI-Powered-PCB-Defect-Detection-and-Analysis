import pandas as pd
from sklearn.model_selection import train_test_split
import os
import tqdm
import shutil

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("./GRAZPEDWRI-DX/dataset.csv")
    
    # Stratified splitting
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["class"], random_state=42
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["class"], random_state=42
    )

    # Save split datasets
    train_df.to_csv("./GRAZPEDWRI-DX/train_data.csv", index=False)
    valid_df.to_csv("./GRAZPEDWRI-DX/valid_data.csv", index=False)
    test_df.to_csv("./GRAZPEDWRI-DX/test_data.csv", index=False)

    # Directories for data organization
    img_dir = "./GRAZPEDWRI-DX/data/images/"
    ann_dir = "./GRAZPEDWRI-DX/data/labels/"
    img_train_dir = "./GRAZPEDWRI-DX/data/images/train/"
    img_valid_dir = "./GRAZPEDWRI-DX/data/images/valid/"
    img_test_dir = "./GRAZPEDWRI-DX/data/images/test/"
    ann_train_dir = "./GRAZPEDWRI-DX/data/labels/train/"
    ann_valid_dir = "./GRAZPEDWRI-DX/data/labels/valid/"
    ann_test_dir = "./GRAZPEDWRI-DX/data/labels/test/"
    
    # Create directories
    for dir in [img_train_dir, img_valid_dir, img_test_dir, ann_train_dir, ann_valid_dir, ann_test_dir]:
        os.makedirs(dir, exist_ok=True)

    # Function to move files
    def move_files(dataframe, img_target_dir, ann_target_dir):
        for i in tqdm.tqdm(dataframe.index, total=len(dataframe)):
            filestem = dataframe.loc[i, "filestem"]
            shutil.move(os.path.join(img_dir, filestem + ".jpg"), os.path.join(img_target_dir, filestem + ".jpg"))
            shutil.move(os.path.join(ann_dir, filestem + ".txt"), os.path.join(ann_target_dir, filestem + ".txt"))

    # Move files into respective directories
    move_files(train_df, img_train_dir, ann_train_dir)
    move_files(valid_df, img_valid_dir, ann_valid_dir)
    move_files(test_df, img_test_dir, ann_test_dir)

    # Print statistics
    total_samples = len(df)
    print("Data split completed with class stratification:")
    print(f"  - {len(train_df)} ({100 * len(train_df) / total_samples:.3f}%) samples in the training set.")
    print(f"  - {len(valid_df)} ({100 * len(valid_df) / total_samples:.3f}%) samples in the validation set.")
    print(f"  - {len(test_df)} ({100 * len(test_df) / total_samples:.3f}%) samples in the testing set.")

