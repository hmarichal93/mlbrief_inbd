import cv2
import numpy as np
import shutil

from pathlib import Path

from lib.image import resize_image_using_pil_lib, Color


class INBD:
    """
    Resize INBD dataset to a fixed size for all images
    """

    def __init__(self, dataset_dir="/data/maestria/datasets/INBD/EH", output_dir=None, size=640):
        self.dataset_dir = dataset_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.images_dir = dataset_dir + "/inputimages"
        self.annotations_dir = dataset_dir + "/annotations"
        self.output_images_dir = self.output_dir / "inputimages"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_annotations_dir = self.output_dir / "annotations"
        self.output_annotations_dir.mkdir(parents=True, exist_ok=True)
        self.debug = True

    def resize_images(self):
        images = list(Path(self.images_dir).rglob("*.jpg"))
        for image in images:
            image = str(image)
            img = cv2.imread(image)
            img = resize_image_using_pil_lib(img, self.size, self.size)
            cv2.imwrite(str(self.output_images_dir / Path(image).name), img)

    def resize_annotations(self):
        import PIL
        from PIL import Image
        annotations = list(Path(self.annotations_dir).rglob("*.tiff"))
        for annotation in annotations:
            annotation = str(annotation)
            ann_img = cv2.imread(annotation, cv2.IMREAD_UNCHANGED)
            debug_dir = "./output"
            Path(debug_dir).mkdir(parents=True, exist_ok=True)

            ring_boundary_id = 0
            mask = (ann_img == ring_boundary_id).astype(np.uint8)
            mask *= 255

            boundary_mask_r = resize_image_using_pil_lib(mask, self.size, self.size, mode=PIL.Image.NEAREST)
            # apply dilatation operation to the mask
            kernel = np.ones((5, 5), np.uint8)
            boundary_mask_r = cv2.dilate(boundary_mask_r, kernel, iterations=2)

            final_ann = resize_image_using_pil_lib(ann_img, self.size, self.size, mode=PIL.Image.NEAREST)
            # save final_ann array using Pil
            final_ann = final_ann.astype(np.int32)
            final_ann[boundary_mask_r > 0] = 0
            pil_image = Image.fromarray(final_ann)
            pil_image.save(str(self.output_annotations_dir / Path(annotation).name))

            # count frequency of each pixel value for final_ann array
            if self.debug:
                # save images
                h, w = final_ann.shape
                debug_img = np.zeros((h, w, 3), dtype=np.uint8)
                debug_img[final_ann == -1] = 255
                color = Color()
                for i in range(1, ann_img.max()):
                    debug_img[final_ann == i] = color.get_next_color()
                debug_img[final_ann == 0] = color.get_next_color()
                cv2.imwrite(self.output_annotations_dir /
                            f"{Path(annotation).stem}.jpg",
                            debug_img)

    def resize_dataset(self):
        self.resize_images()
        self.resize_annotations()
        # copy the .txt in root directory
        txt_files = list(Path(self.dataset_dir).rglob("*.txt"))
        for txt_file in txt_files:
            txt_file = str(txt_file)
            txt_file_name = Path(txt_file).name
            txt_file_output = self.output_dir / txt_file_name
            shutil.copy(txt_file, txt_file_output)

        return


def main(dataset_dir, output_folder, size):
    dataset = INBD( dataset_dir = dataset_dir,
                    output_dir = output_folder,
                    size = size )
    dataset.resize_dataset()

    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Resize INBD dataset')
    parser.add_argument('--dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--size', type=int, help='Output image size')
    args = parser.parse_args()

    main(args.dataset_dir, args.output_folder, args.size)


