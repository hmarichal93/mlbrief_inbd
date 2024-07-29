import numpy as np
import cv2
from pathlib import Path
import json
from fpdf import FPDF

from lib.image import Color
from lib.utils import polygon_2_labelme_json
from lib.io import write_json

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


    def transform_inbd_labelmap_to_contours(self, image_path, root_inbd_results):
        image_name = Path(image_path).name
        inbd_labelmap_path = f'{root_inbd_results}/{image_name}.labelmap.npy'
        if not Path(inbd_labelmap_path).exists():
            return [], None

        image = cv2.imread(image_path)
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

            contours_list.append(contours[0].squeeze())
            # draw contours on image
            if self.debug:
                img_contour = image.copy()
                img_contour[region_mask] = color.get_next_color()
                cv2.drawContours(img_contour, contours, -1, color.get_next_color(), 3)
                cv2.drawContours(image_debug, contours, -1, color.get_next_color(), 3)
                #cv2.imwrite(f'{output_dir}/contour_{region}.png', img_contour)

        if self.debug:
            cv2.imwrite(f'{output_dir}/contours.png', image_debug)
            cv2.imwrite(f'{output_dir}/image.png', image)

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
            "resultados/2024-07-23_17h03m29s_INBD_100e_a6.3__"
         , output_dir = "./output"):
    debug = True
    conversor = FromINBD2UruDendro(output_dir = output_dir, debug=debug)


    images_path = [Path(root_dataset)] if Path(root_dataset).is_file() else Path(f"{root_dataset}/InputImages").rglob("*.jpg")
    for image_path in images_path:
        conversor.transform_inbd_labelmap_to_contours(image_path, root_inbd_results)
        contours, image = conversor.transform_inbd_labelmap_to_contours(image_path, root_inbd_results)
        if len(contours)==0:
            continue
        H, W, _ = image.shape
        image_name = Path(image_path).name
        image_stem = image_name.split(".")[0]
        labelme_json = conversor.from_contour_to_urudendro(contours,  H, W, image_path,
                                                save_path = f'{conversor.output_dir}/{image_stem}/{image_stem}.json')

    if debug:
        conversor.generate_pdf()



    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dataset", type=str, default="/data/maestria/datasets/Candice_inbd_1500/")
    parser.add_argument("--root_inbd_results", type=str, default="/data/maestria/resultados/inbd_pinus_taeda_1500/candice_transfer_learning/"
            "resultados/2024-07-23_17h03m29s_INBD_100e_a6.3__")
    parser.add_argument("--output_dir", type=str, default="./output")

    args = parser.parse_args()
    main(args.root_dataset, args.root_inbd_results, args.output_dir)

