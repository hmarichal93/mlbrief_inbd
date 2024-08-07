import os

import numpy as np
import pandas as pd
import argparse
import cv2
import json

from pathlib import Path
from shapely.geometry import Polygon, Point

from uruDendro.metric_influence_area import  main as urudendro_metric
from uruDendro.lib import drawing as dr
def load_json(filepath: str) -> dict:
    """
    Load json utility.
    :param filepath: file to json file
    :return: the loaded json as a dictionary
    """
    with open(str(filepath), 'r') as f:
        data = json.load(f)
    return data

def load_ring_stimation(path):
    try:
        json_content = load_json(path)
        l_rings = []
        for ring in json_content['shapes']:
            l_rings.append(Polygon(np.array(ring['points'])[:, [1, 0]].tolist()))

    except FileNotFoundError:
        l_rings = []

    return l_rings
def main(annotations_file_path = "/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/test_and_val_annotations.txt",
         root_original_dataset = "/data/maestria/resultados/mlbrief_inbd/PinusTaedaV1",
         output_dir = "/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/inbd_results/inbd_/inbd_urudendro_labels_original_shape",
         inbd_inference_results_dir = "/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/inbd_results/inbd_/inbd_urudendro_labels",
         inbd_center_mask_dir = "/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/center"):

    annotations_original_dataset_dir = f"{root_original_dataset}/anotaciones/labelme/images"
    df_annotations = pd.read_csv(annotations_file_path, header=None)
    images_path = Path(root_original_dataset) / "images/segmented"


    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for idx, row in df_annotations.iterrows():
        annotation_path = row.iloc[0]
        center_mask_path = Path(inbd_center_mask_dir) / f"{Path(annotation_path).stem}.png"
        center_mask = cv2.imread( str(center_mask_path), cv2.IMREAD_UNCHANGED)
        #get mean of the center mask
        cx, cy = np.argwhere(center_mask).mean(0).tolist()

        df_filename_path = Path(inbd_inference_results_dir).rglob(f"{Path(annotation_path).stem}.json")
        df_filename_path = [path for path in df_filename_path]
        if list(df_filename_path) == 0:
            continue

        df_filename_inbd = list(df_filename_path)[0]
        #load json file
        df = load_json(str(df_filename_inbd))
        image_resize_path = df["imagePath"]
        image_resize = cv2.imread(image_resize_path, cv2.IMREAD_GRAYSCALE)
        image_mask = image_resize < 255
        image_mask = image_mask.astype(np.uint8) * 255
        h, w = center_mask.shape

        gt_filename = Path(annotations_original_dataset_dir) / f"{Path(annotation_path).stem}.json"
        df_gt = load_json(str(gt_filename))
        H = df_gt["imageHeight"]
        W = df_gt["imageWidth"]


        #get the rings
        rings = load_ring_stimation(df_filename_inbd)
        l_list = []
        for idx, r in enumerate(rings):
            y, x = r.exterior.coords.xy
            if not r.contains(Point(cy,cx)):
                continue
            #if all the pixels defined in y,x are outside the image mask (image_mask), then skip this ring
            if all([image_mask[int(yi), int(xi)] == 0 for yi, xi in zip(y, x)]):
                continue
            x = np.array(x) * W / w
            y = np.array(y) * H / h

            dictionary = {}
            dictionary["label"] = str(idx + 1)
            dictionary["shape_type"] = "polygon"
            dictionary["points"] = np.array([x, y]).T.tolist()
            dictionary["flags"] = {}
            l_list.append(dictionary)

        cy = cy * H / h
        cx = cx * W / w
        ################################################################################################################
        img_filename = str(images_path.glob(f"{Path(annotation_path).stem}*").__next__())
        labelme_json = {"imagePath": img_filename,
                        "imageHeight": H,
                        "imageWidth": W,
                        "version": "5.0.1",
                        "flags": {},
                        "shapes": l_list,
                        "imageData": None,
                        'exec_time(s)': 0,
                        'center': [cy , cx ]
        }

        (Path(output_dir) / Path(annotation_path).stem).mkdir(parents=True, exist_ok=True)
        dt_output_path = Path(output_dir) / Path(annotation_path).stem / f"{Path(annotation_path).stem}.json"
        with open(str(dt_output_path), 'w') as f:
            json.dump(labelme_json, f, indent=4)

        ################################################################################################################

        output_dir_image  = Path(output_dir) / Path(annotation_path).stem

        disk_name = Path(annotation_path).stem
        print(f"Processing {disk_name}")
        # if not disk_name == 'F02a':
        #     continue
        P, R, F, RMSE, TP, FP, TN, FN = urudendro_metric(dt_file=dt_output_path, gt_file=gt_filename,
                                    img_filename=img_filename, cx=cx, cy=cy, output_dir=output_dir_image, threshold=0.6)











    return


if __name__ == "__main__":
    main()