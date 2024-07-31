import numpy as np
import cv2
import pandas as pd
from pathlib import Path

def generate_annotation_mask(annotation_path:str, output_dir:str):
    """Generate annotation mask from annotation_path and save it in output_dir"""
    annotation = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
    ring_boundary_id = 1
    mask = (annotation == ring_boundary_id).astype(np.uint8)
    mask *= 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    output_path = Path(output_dir) /  f"{Path(annotation_path).stem}.png"
    cv2.imwrite(str(output_path), mask)
    return


def main(input_images_path, input_annotations_path, root_dataset, output_dir, inbd_model_path):
    #1.0 Generate center mask images
    df_annotations = pd.read_csv(input_annotations_path, header=None)
    center_mask_dir = Path(output_dir) / "center"
    center_mask_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in df_annotations.iterrows():
        annotation_path = row.iloc[0]
        annotation_path = Path(root_dataset) / annotation_path
        generate_annotation_mask( str(annotation_path), str(center_mask_dir))

    # 2.0 run INBD inference
    # 2.1 get script file path
    root_inbd = Path(__file__).parent.parent / "INBD"
    inbd_path = root_inbd / "main.py"
    # 2.2 run INBD inference
    df_images = pd.read_csv(input_images_path, header=None)
    inbd_results_dir = Path(output_dir) / "inbd_results"
    import subprocess
    for idx, row in df_images.iterrows():
        image_path = Path(root_dataset) / row.iloc[idx]
        center_mask_path = center_mask_dir /  f"{Path(image_path).stem}.png"
        cmd = f"python  {inbd_path} inference {inbd_model_path} {image_path} {center_mask_path} --output {inbd_results_dir}"
        subprocess.run(cmd, shell=True)

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Resize INBD dataset')
    parser.add_argument('--input_images_path', type=str, help='Input images path')
    parser.add_argument('--input_annotations_path', type=str, help='Input annotations path')
    parser.add_argument('--root_dataset', type=str, help='Root dataset')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--inbd_model_path', type=str, help='INBD model path')

    args = parser.parse_args()
    main(args.input_images_path, args.input_annotations_path, args.root_dataset, args.output_dir, args.inbd_model_path)