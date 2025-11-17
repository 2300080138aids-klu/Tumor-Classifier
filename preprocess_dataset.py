import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

def preprocess_braTS(
    src_root=r"F:\MRI\Backend\Dataset\TrainingData", 
    dst_root="dataset",
    min_tumor_pixels=50
):
    """
    Converts BraTS .nii files to 2D PNG slices and classifies them into
    tumor / no_tumor using the segmentation mask.
    """

    tumor_dir = os.path.join(dst_root, "tumor")
    no_tumor_dir = os.path.join(dst_root, "no_tumor")
    os.makedirs(tumor_dir, exist_ok=True)
    os.makedirs(no_tumor_dir, exist_ok=True)

    # Loop through all patient folders
    for patient in tqdm(os.listdir(src_root), desc="Processing patients"):
        p_path = os.path.join(src_root, patient)
        if not os.path.isdir(p_path):
            continue

        # Find the modality and segmentation files
        flair_path = next((os.path.join(p_path, f) for f in os.listdir(p_path) if "flair" in f and f.endswith(".nii")), None)
        seg_path = next((os.path.join(p_path, f) for f in os.listdir(p_path) if "seg" in f and f.endswith(".nii")), None)

        if not flair_path or not seg_path:
            print(f"[WARN] Missing flair or seg for {patient}")
            continue

        # Load volumes
        flair_img = nib.load(flair_path).get_fdata()
        seg_img = nib.load(seg_path).get_fdata()

        # Normalize intensity
        flair_img = (flair_img - np.min(flair_img)) / (np.max(flair_img) - np.min(flair_img) + 1e-8)

        num_slices = flair_img.shape[2]

        for i in range(num_slices):
            flair_slice = (flair_img[:, :, i] * 255).astype(np.uint8)
            mask_slice = seg_img[:, :, i]

            # Determine if this slice has tumor
            has_tumor = np.count_nonzero(mask_slice) > min_tumor_pixels
            target_dir = tumor_dir if has_tumor else no_tumor_dir

            # Resize for consistency
            flair_slice = cv2.resize(flair_slice, (128, 128))

            # Save slice
            out_name = f"{patient}_slice_{i:03d}.png"
            out_path = os.path.join(target_dir, out_name)
            cv2.imwrite(out_path, flair_slice)

    print("✅ Done!")
    print(f"Tumor slices: {len(os.listdir(tumor_dir))}")
    print(f"No-tumor slices: {len(os.listdir(no_tumor_dir))}")


if __name__ == "__main__":
    preprocess_braTS()
