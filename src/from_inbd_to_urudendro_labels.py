import numpy as np
import cv2
from pathlib import Path
import json
from fpdf import FPDF
from shapely.geometry import Polygon, Point

from lib.image import Color
from lib.utils import polygon_2_labelme_json
from lib.io import write_json
from urudendro import drawing as dr
class FromINBD2UruDendro:
    def __init__(self, output_dir=None, debug = True):
        self.debug = debug

        if output_dir is None:
            raise ValueError("output_dir must be defined")
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


    def load_inbd_labelmap(self, inbd_labelmap_path):
        inbd_labelmap = np.load(inbd_labelmap_path)
        return inbd_labelmap

    def make_contour_of_thickness_one(self, contour, inbd_labelmap, output_dir=None):
        mask = np.zeros_like(inbd_labelmap)
        mask = dr.Drawing.curve(Polygon(contour[:,[1,0]].tolist()).exterior, mask, 255, 1)
        if output_dir:
            cv2.imwrite(f'{output_dir}/mask.png', mask)
        mask = mask > 0
        # apply skeleton operation over mask skimage
        from skimage.morphology import skeletonize
        # Apply skeleton operation over mask using skimage
        mask = skeletonize(mask.astype(np.uint8))  # get contours of mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0].squeeze()
        if output_dir:
            mask = np.zeros_like(inbd_labelmap)
            mask = dr.Drawing.curve(Polygon(contour[:, [1, 0]].tolist()).exterior, mask, 255, 1)
            cv2.imwrite(f'{output_dir}/skeleton.png', mask)

        return contour

    def transform_inbd_labelmap_to_contours(self, image_path, center_mask_path, root_inbd_results, minimum_pixels=50):
        image_name = Path(image_path).stem
        label_name = f"{image_name}*"
        inbd_labelmap_path = [  path for path in  Path(root_inbd_results).glob(label_name) if "labelmap.npy" in path.name]
        if len(inbd_labelmap_path)==0:
            print(inbd_labelmap_path)
            return [], None
        inbd_labelmap_path = inbd_labelmap_path[0]
        if not Path(center_mask_path).exists():
            raise "center mask not found"
        center_mask = cv2.imread(center_mask_path, cv2.IMREAD_UNCHANGED)
        cy, cx = np.argwhere(center_mask).mean(0).tolist()


        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image {image_path} not found")
        image_debug = image.copy()
        inbd_labelmap = self.load_inbd_labelmap(inbd_labelmap_path)
        region_ids = np.unique(inbd_labelmap)
        #remove background region (id=0)
        region_ids = region_ids[region_ids > 0]
        contours_list = []
        if self.debug:
            output_dir = Path(self.output_dir) / Path(image_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            color = Color()
        mask = np.zeros_like(inbd_labelmap)
        for region in region_ids:
            region_mask = inbd_labelmap == region
            mask[region_mask] = 255
            # get contours of region
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            #Contour must have a thickness == 1
            contour = contours[0].squeeze()
            if contour.ndim == 1:
                continue
            if contour.shape[0] < minimum_pixels:
                continue

            #contour = self.make_contour_of_thickness_one(contour, inbd_labelmap)#, output_dir)
            contour_poly = Polygon(contour[:, [1, 0]].tolist())
            if not contour_poly.contains(Point(cy,cx)):
                continue

            contours_list.append(contour)
            # draw contours on image
            if self.debug:
                img_contour = image.copy()
                img_contour[region_mask] = color.get_next_color()
                cv2.drawContours(img_contour, contours, -1, color.get_next_color(), 3)
                cv2.drawContours(image_debug, contours, -1, color.get_next_color(), 3)
                #cv2.imwrite(f'{output_dir}/contour_{region}.png', img_contour)

        if self.debug:
            cv2.imwrite(f'{output_dir}/contours.png', image_debug)
            print(image_debug)
            cv2.imwrite(f'{output_dir}/image.png', image)
            print(f"Contour images are stored in {output_dir}")
            print([filename for filename in Path(output_dir).glob("*.png")])


        return contours_list, image

    def from_contour_to_urudendro(self, chain_list, image_height, image_width,  image_path, save_path=None):
            """
            Converting ch_i list object to labelme format. This format is used to store the coordinates of the rings at the image
            original resolution
            @param chain_list: ch_i list
            @param image_path: image input path
            @param image_height: image hegith
            @param image_width: image width_output
            @param img_orig: input image
            @param exec_time: method execution time
            @param cy: pith y's coordinate
            @param cx: pith x's coordinate
            @return:
            - labelme_json: json in labelme format. Ring coordinates are stored here.
            """

            labelme_json = polygon_2_labelme_json(chain_list, image_height, image_width, image_path )

            if save_path:
                write_json(labelme_json, str(save_path))

            return labelme_json

    def generate_pdf(self):
        pdf = FPDF()
        images_path = Path(self.output_dir).rglob("image.png")
        for image in images_path:
            name = image.parent.stem
            pdf.add_page()
            #print name in the top left corner
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=name, ln=True, align='L')
            pdf.image(str(image), x=10, y=20, w=200)
            pdf.add_page()

            contour = image.parent / "contours.png"
            pdf.image(str(contour), x=10, y=20, w=200)

        pdf.output(f"{self.output_dir}/output.pdf")


def main(root_dataset = "/data/maestria/datasets/Candice_inbd_1500/",
         root_inbd_results = "/data/maestria/resultados/inbd_pinus_taeda_1500/candice_transfer_learning/"
            "resultados/2024-07-23_17h03m29s_INBD_100e_a6.3__",
         center_mask_dir = "/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/center"
         , output_dir = "./output"):
    debug = True

    output_dir = Path(output_dir) / "inbd_urudendro_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)

    conversor = FromINBD2UruDendro(output_dir = output_dir, debug=debug)

    images_path = [Path(root_dataset)] if Path(root_dataset).is_file() else Path(f"{root_dataset}/InputImages").rglob("*.jpg")
    for image_path in images_path:
        center_mask_path = Path(center_mask_dir) / (Path(image_path).stem + ".png") if Path(center_mask_dir).is_dir()\
            else center_mask_dir
        contours, image = conversor.transform_inbd_labelmap_to_contours( str(image_path), center_mask_path, root_inbd_results)
        if len(contours)==0:
            continue
        H, W, _ = image.shape
        image_name = Path(image_path).name
        image_stem = image_name.split(".")[0]
        output_dir_image = Path(output_dir) / image_stem
        output_dir_image.mkdir(parents=True, exist_ok=True)

        #print(image_label_path)
        image_label_path = f'{str(output_dir_image)}/{image_stem}.json'
        labelme_json = conversor.from_contour_to_urudendro(contours,  H, W, image_path,
                                                save_path = image_label_path)

    if debug:
        conversor.generate_pdf()

    print(f"Labels are stored in {output_dir_image}")

    return output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dataset", type=str, default="/data/maestria/datasets/Candice_inbd_1500/")
    parser.add_argument("--root_inbd_results", type=str,
                        default="/data/maestria/resultados/inbd_pinus_taeda_1500/candice_transfer_learning/"
                          "resultados/2024-07-23_17h03m29s_INBD_100e_a6.3__")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--center_mask_dir", type=str,
                        default="/data/maestria/resultados/mlbrief_PinusTaedaV1_1500/inference/center)")

    args = parser.parse_args()
    main(root_dataset= args.root_dataset, root_inbd_results = args.root_inbd_results, output_dir=  args.output_dir,
         center_mask_dir= args.center_mask_dir)