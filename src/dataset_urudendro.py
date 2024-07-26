import cv2
import numpy as np
import shutil
import pandas as pd
import os
from skimage import io

from shapely.geometry import Polygon
from pathlib import Path
from torch.utils.data import Dataset

from lib.image import resize_image_using_pil_lib, Color
from lib.io import load_json
from dataset_inbd import InspectAnnotations


class TreeRingDataloader(Dataset):
    """
    Dataloader for dataset formatted to use with INBD
    """
    def __init__(self, annotation_csv_file, images_csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(annotation_csv_file, header=None)
        self.images = pd.read_csv(images_csv_file, header=None)
        self.annotations_path = annotation_csv_file
        self.images_path = images_csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):

        image_path = os.path.join(self.root_dir, self.images.iloc[item, 0])
        image = io.imread(image_path)
        label_path = os.path.join(self.root_dir, self.annotations.iloc[item, 0])
        label_path = label_path.replace(".tiff", ".png")
        label = io.imread(label_path)
        disk_name = Path(self.annotations.iloc[item, 0]).stem
        if self.transform:
            image = self.transform(image)

        return (image, image_path, label, label_path,  disk_name)


class PinusTaeda:
    def __init__(self, dataset_dir="/data/maestria/resultados/inbd_pinus_taeda/pinus_teada/",
                 output_dir=None):
        self.dataset_dir = dataset_dir
        Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)
        self.images_dir = dataset_dir + "/images"
        self.annotations_dir = dataset_dir + "/anotaciones/labelme/images"
        self.output_dir = Path(output_dir)
        self.output_annotations_dir = self.output_dir / "annotations"
        self.output_annotations_dir.mkdir(parents=True, exist_ok=True)
        self.output_images_dir = self.output_dir / "InputImages"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)

        self.train_images_path = None
        self.val_images_path = None
        self.test_images_path = None

        self.train_annotations_path = None
        self.val_annotations_path = None
        self.test_annotations_path = None

    def __get_minimum_ring_area_index(self, annotation, img=None, area_threshold=0.005):
        l_rings = self.load_ring_stimation(annotation)
        l_rings.sort(key=lambda x: x.area, reverse=False)

        area_image = img.shape[0] * img.shape[1]
        for i, ring in enumerate(l_rings):
            if ring.area > area_image * area_threshold:
                index = i
                break
        index = 1
        return index

    def transform_annotations(self, size=None):
        """
        Transform annotations from labelme to mask. Save the mask in the output directory
        :return:
        """

        self.input_images_path = self.output_dir / "train_inputimages.txt"
        self.annotations_path = self.output_dir / "train_annotations.txt"

        if self.input_images_path.exists() and self.annotations_path.exists():
            return

        annotations = list(Path(self.annotations_dir).rglob("*.json"))
        l_masks_path = []
        l_images_path = []
        for annotation in annotations:
            annotation = str(annotation)
            img_name = Path(annotation).name.split('.')[0]
            img_path = Path(self.images_dir).rglob(f"segmented/{img_name}.*").__next__()  # get the image path
            img = cv2.imread(str(img_path))

            # pith_size = 3 if disk_name in disk_with_small_pith else 3
            # pith_size = self.__get_minimum_ring_area_index(annotation, img)
            pith_size = 1
            segmentation_mask, boundaries_mask = self.annotation_to_mask(annotation, img, pith_size=pith_size,
                                                                         size=size)
            if size:
                img = resize_image_using_pil_lib(img, size, size)
            segmentation_mask_name = Path(annotation).name.replace(".json", ".png")
            mask_path = self.output_annotations_dir / segmentation_mask_name
            cv2.imwrite(str(mask_path), segmentation_mask)

            bounderies_mask_name = Path(annotation).name.replace(".json", ".tiff")
            bounderies_mask_path = self.output_annotations_dir / bounderies_mask_name
            # cv2.imwrite(str(bounderies_mask_path), boundaries_mask)

            ### save image
            img_output_name = Path(img_path).name.split('.')[0] + ".jpg"
            img_output_path = self.output_images_dir / img_output_name
            cv2.imwrite(str(img_output_path), img)

            ###

            from PIL import Image
            # Convert the numpy array to a PIL.Image object
            boundaries_mask_pil = Image.fromarray(boundaries_mask.astype(np.int32), mode='I')

            # Save the image

            boundaries_mask_pil.save(str(bounderies_mask_path))

            get_relative_path = lambda x: x.relative_to(self.output_dir)

            l_images_path.append(str(get_relative_path(img_output_path)))
            l_masks_path.append(str(get_relative_path(bounderies_mask_path)))

        # save the paths as txt
        self.save_as_txt(self.input_images_path, l_images_path)
        self.save_as_txt(self.annotations_path, l_masks_path)
        return

    @staticmethod
    def save_as_txt(output_path, l_objects):
        with open(output_path, "w") as f:
            f.write("\n".join(l_objects))

    @staticmethod
    def read_txt(path):
        with open(path, "r") as f:
            l_objects = f.readlines()

        l_objects = [item.replace("\n", "") for item in l_objects]
        return l_objects

    @staticmethod
    def load_ring_stimation(path):
        try:
            json_content = load_json(path)
            l_rings = []
            for ring in json_content['shapes']:
                l_rings.append(Polygon(np.array(ring['points'])[:, [0, 1]].tolist()))

        except FileNotFoundError:
            l_rings = []

        l_rings.sort(key=lambda x: x.area, reverse=False)
        return l_rings

    def annotation_to_mask(self, annotation, img, size=None, pith_size=0):
        """
        Transform annotation to mask
        :param annotation: annotation path
        :return: mask
        """

        l_rings = self.load_ring_stimation(annotation)

        if size:
            H, W, _ = img.shape
            img = resize_image_using_pil_lib(img, size, size)
            h_s, w_s, _ = img.shape
            # resize rings
            l_rings_aux = []
            for ring in l_rings:
                l_rings_aux.append(Polygon(list(np.array(ring.exterior.coords) * [w_s / W, h_s / H])))

            l_rings = l_rings_aux

        # 1.0 create mask
        segmentation_mask = np.zeros_like(img, dtype=np.uint8)
        # background equal to -1
        boundaries_mask = np.zeros(img.shape[:2], dtype=np.int8) + -1

        color = Color()
        boundaries_thickness = 3

        # 2.0 fill mask
        for i, ring in enumerate(l_rings):
            # 2.2 draw area
            if i < pith_size + 1:
                # fill poly
                cv2.fillPoly(boundaries_mask, [np.array(ring.exterior.coords, dtype=np.int32)], 1)
                # 2.3 draw ring boundaries
                cv2.polylines(boundaries_mask, [np.array(ring.exterior.coords, dtype=np.int32)], isClosed=True, color=0,
                              thickness=boundaries_thickness)

            else:
                # fill area between ring i and ring i - 1
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(ring.exterior.coords, dtype=np.int32)], 1)
                cv2.fillPoly(mask, [np.array(l_rings[i - 1].exterior.coords, dtype=np.int32)], 2)
                # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask[mask == 1] = 255
                mask[mask == 2] = 0
                boundaries_mask[mask == 255] = i + 1
                # 2.3 draw ring boundaries
                cv2.polylines(boundaries_mask, [np.array(ring.exterior.coords, dtype=np.int32)], isClosed=True, color=0,
                              thickness=boundaries_thickness)

        l_rings.sort(key=lambda x: x.area, reverse=True)
        for i, ring in enumerate(l_rings):
            if i < len(l_rings) - pith_size:
                cv2.fillPoly(segmentation_mask, [np.array(ring.exterior.coords, dtype=np.int32)],
                             color.get_next_color())
                cv2.polylines(segmentation_mask, [np.array(ring.exterior.coords, dtype=np.int32)],
                              isClosed=True, color=0, thickness=boundaries_thickness)

        return segmentation_mask, boundaries_mask

    def split_dataset_in_train_val_and_test(self, val_size=0.2, test_size=0.2):
        """
        Split the dataset in train, validation and test
        :param val_size: validation size
        :param test_size: test size
        :return:
        """
        self.train_images_path = self.output_dir / "train_images.txt"
        self.val_images_path = self.output_dir / "val_images.txt"
        self.test_images_path = self.output_dir / "test_images.txt"

        self.train_annotations_path = self.output_dir / "train_annotations.txt"
        self.val_annotations_path = self.output_dir / "val_annotations.txt"
        self.test_annotations_path = self.output_dir / "test_annotations.txt"

        if self.train_images_path.exists() and self.val_images_path.exists() and self.test_images_path.exists():
            return

        l_images = self.read_txt(self.input_images_path)
        l_annotations = self.read_txt(self.annotations_path)

        # suffle the dataset
        np.random.seed(42)
        lenght = len(l_images)
        # generate a list from 0 to lenght
        l_indexes = np.arange(lenght)
        np.random.shuffle(l_indexes)
        # split the dataset
        val_index = int(lenght * val_size)
        test_index = int(lenght * test_size)
        train_index = lenght - val_index - test_index

        train_images_index = l_indexes[:train_index]
        val_images_index = l_indexes[train_index:train_index + val_index]
        test_images_index = l_indexes[train_index + val_index:]

        self.save_as_txt(self.train_images_path, [l_images[i] for i in train_images_index])
        self.save_as_txt(self.val_images_path, [l_images[i] for i in val_images_index])
        self.save_as_txt(self.test_images_path, [l_images[i] for i in test_images_index])

        self.save_as_txt(self.train_annotations_path, [l_annotations[i] for i in train_images_index])
        self.save_as_txt(self.val_annotations_path, [l_annotations[i] for i in val_images_index])
        self.save_as_txt(self.test_annotations_path, [l_annotations[i] for i in test_images_index])

        return

    def create_dataloaders(self):
        train = TreeRingDataloader(annotation_csv_file=self.train_annotations_path,
                                   images_csv_file=self.train_images_path,
                                   root_dir=self.output_dir)
        val = TreeRingDataloader(annotation_csv_file=self.val_annotations_path, images_csv_file=self.val_images_path,
                                 root_dir=self.output_dir)
        test = TreeRingDataloader(annotation_csv_file=self.test_annotations_path, images_csv_file=self.test_images_path,
                                  root_dir=self.output_dir)

        return train, val, test

    def convert_numpy_image_to_pil_image(self, image):
        """
        Convert numpy image to PIL image
        :param image: numpy image
        :return: PIL image
        """
        from PIL import Image
        return Image.fromarray(image)

    def reshape_using_pil_lib(self, pil_img, height_output, width_output, keep_ratio=True):
        """
        Resize image using PIL library.
        @param im_in: input_image image
        @param height_output: output image height_output
        @param width_output: output image width_output
        @return: matrix with the resized image
        """
        from PIL import Image
        # Image.ANTIALIAS is deprecated, PIL recommends using Reampling.LANCZOS
        # flag = Image.ANTIALIAS
        flag = Image.Resampling.LANCZOS
        if keep_ratio:
            aspect_ratio = pil_img.height / pil_img.width
            if pil_img.width > pil_img.height:
                height_output = int(width_output * aspect_ratio)
            else:
                width_output = int(height_output / aspect_ratio)

        pil_img = pil_img.resize((width_output, height_output), flag)

        return pil_img

    def visualization(self, dataloader: TreeRingDataloader, pdf_output=None, shape=640):
        """
        Display image and label
        :param dataloader: dataloader
        :param index: index
        :return:
        """
        images = []
        # generate pdf
        for idx in range(dataloader.__len__()):
            image, annotation, disk_name = dataloader.__getitem__(idx)
            image = self.convert_numpy_image_to_pil_image(image)
            image_r = self.reshape_using_pil_lib(image, shape, shape)
            images.append(image_r)
            annotation = self.convert_numpy_image_to_pil_image(annotation)
            annotation_r = self.reshape_using_pil_lib(annotation, shape, shape)
            images.append(annotation_r)

        images[0].save(pdf_output, save_all=True, append_images=images[1:], quality=100)

    from pathlib import Path



def build_dataset(dataset_dir, output_dir='/data/maestria/resultados/inbd_2', size=None):
    dataset = PinusTaeda(dataset_dir=dataset_dir, output_dir=output_dir)
    dataset.transform_annotations(size=size)
    dataset.split_dataset_in_train_val_and_test()
    train_dataloader, val_dataloader, test_dataloader = dataset.create_dataloaders()
    train_pdf_path = Path(output_dir) / 'train.pdf'
    dataset.visualization(train_dataloader, pdf_output=train_pdf_path)
    val_pdf_path = Path(output_dir) / 'val.pdf'
    dataset.visualization(val_dataloader, pdf_output=val_pdf_path)
    test_pdf_path = Path(output_dir) / 'test.pdf'
    dataset.visualization(test_dataloader, pdf_output=test_pdf_path)

    return

def inspect_annotations(dataset_dir, output_dir='/data/maestria/resultados/inbd/inspect_EH'):
    dataset = InspectAnnotations(dataset_dir=dataset_dir, output_dir=output_dir)
    dataset.inspect_tiff_annotations()
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('--output_folder', type=str, help='Output folder')

    parser.add_argument('--size', type=int, help='Output image size')
    ##resize flag
    args = parser.parse_args()
    build_dataset(args.dataset_dir, args.output_folder, args.size)
    inspect_annotations(dataset_dir=args.output_folder,
                        output_dir=Path(args.output_folder) / 'inspect')