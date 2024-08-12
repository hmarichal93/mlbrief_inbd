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

def get_center_pixel(annotation_path, inbd_center_mask_dir):
    center_mask_path = Path(inbd_center_mask_dir) / f"{Path(annotation_path).stem}.png"
    center_mask = cv2.imread( str(center_mask_path), cv2.IMREAD_UNCHANGED)
    #get mean of the center mask
    cx, cy = np.argwhere(center_mask).mean(0).tolist()
    return cx, cy
def get_inbd_detection_path(annotation_path, inbd_inference_results_dir):
    df_filename_path = Path(inbd_inference_results_dir).rglob(f"{Path(annotation_path).stem}.json")
    df_filename_path = [path for path in df_filename_path]
    if list(df_filename_path) == 0:
        return None
    return list(df_filename_path)[0]

def get_image_mask(df_filename_inbd):
    df = load_json(str(df_filename_inbd))
    image_resize_path = df["imagePath"]
    image_resize = cv2.imread(image_resize_path, cv2.IMREAD_GRAYSCALE)
    image_mask = image_resize < 255
    image_mask = image_mask.astype(np.uint8) * 255
    return image_mask

def get_original_image_size(annotations_original_dataset_dir, annotation_path):
    gt_filename = Path(annotations_original_dataset_dir) / f"{Path(annotation_path).stem}.json"
    df_gt = load_json(str(gt_filename))
    H = df_gt["imageHeight"]
    W = df_gt["imageWidth"]
    return H, W, gt_filename


def transform_inbd_detection_to_original_image_size(df_filename_inbd, H, W, h, w, cy, cx, image_mask, original_images_dir,
                                                    annotation_path):
    rings = load_ring_stimation(df_filename_inbd)
    l_list = []
    for idx, r in enumerate(rings):
        y, x = r.exterior.coords.xy
        if not r.contains(Point(cx,cy)):
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
    img_filename = str(original_images_dir.glob(f"{Path(annotation_path).stem}*").__next__())
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
    return labelme_json
def main(annotations_file_path = "/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/test_and_val_annotations.txt",
         root_original_dataset = "/data/maestria/resultados/mlbrief_inbd/PinusTaedaV1",
         output_dir = "/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/inbd_results/inbd_/inbd_urudendro_labels_original_shape",
         inbd_inference_results_dir = "/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/inbd_results/inbd_/inbd_urudendro_labels",
         inbd_center_mask_dir = "/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/center"):

    #1)Initializations
    annotations_original_dataset_dir = f"{root_original_dataset}/anotaciones/labelme/images"
    df_annotations = pd.read_csv(annotations_file_path, header=None)
    original_images_dir = Path(root_original_dataset) / "images/segmented"
    rows = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    #2) Loop over the annotations
    for idx, row in df_annotations.iterrows():
        annotation_path = row.iloc[0]
        disk_name = Path(annotation_path).stem

        cx, cy = get_center_pixel(annotation_path, inbd_center_mask_dir)

        df_filename_inbd = get_inbd_detection_path(annotation_path = annotation_path,
                                                   inbd_inference_results_dir=inbd_inference_results_dir)
        if df_filename_inbd is None:
            continue

        image_mask = get_image_mask(df_filename_inbd=df_filename_inbd)
        h, w = image_mask.shape

        H, W, gt_filename = get_original_image_size(annotations_original_dataset_dir=annotations_original_dataset_dir,
                                                    annotation_path=annotation_path)

        # transform the INBD detection to the original image size
        labelme_json = transform_inbd_detection_to_original_image_size(df_filename_inbd=df_filename_inbd,
                                                                       H=H, W=W, h=h, w=w, cy=cy, cx=cx,
                                                                       image_mask=image_mask,
                                                                       original_images_dir=original_images_dir,
                                                                       annotation_path=annotation_path)

        (Path(output_dir) / Path(annotation_path).stem).mkdir(parents=True, exist_ok=True)
        dt_output_path = Path(output_dir) / Path(annotation_path).stem / f"{Path(annotation_path).stem}.json"
        with open(str(dt_output_path), 'w') as f:
            json.dump(labelme_json, f, indent=4)

        ################################################################################################################
        output_dir_image  = Path(output_dir) / Path(annotation_path).stem
        cy, cx = labelme_json["center"]
        P, R, F, RMSE, TP, FP, TN, FN = urudendro_metric(dt_file=dt_output_path, gt_file=gt_filename,
                                    img_filename=labelme_json["imagePath"], cx=cx, cy=cy, output_dir=output_dir_image,
                                    threshold=0.6)

        row = {"image": disk_name , "P": P, "R": R, "F": F, "RMSE": RMSE, "TP": TP,
               "FP": FP, "TN": TN, "FN": FN}
        #add row to dataset but not use append
        rows.append(row)

    df_metric = pd.concat([pd.DataFrame([row]) for row in rows], ignore_index=True)
    df_metric.to_csv(f"{output_dir}/metric_urudendro.csv", index=False)

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_file_path", type=str,
                        default="/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/test_and_val_annotations.txt")
    parser.add_argument("--root_original_dataset", type=str,
                        default="/data/maestria/resultados/mlbrief_inbd/PinusTaedaV1")
    parser.add_argument("--output_dir", type=str, default="/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/inbd_results/inbd_/inbd_urudendro_labels_original_shape")
    parser.add_argument("--inbd_inference_results_dir", type=str,
                        default="/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/inbd_results/inbd_/inbd_urudendro_labels")
    parser.add_argument("--inbd_center_mask_dir", type=str, default="/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/center")

    args = parser.parse_args()
    main(annotations_file_path=args.annotations_file_path, root_original_dataset=args.root_original_dataset,
         output_dir=args.output_dir, inbd_inference_results_dir=args.inbd_inference_results_dir,
         inbd_center_mask_dir=args.inbd_center_mask_dir)