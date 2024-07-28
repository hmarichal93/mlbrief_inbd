import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from lib.io import write_json, load_json
from dataset_urudendro import TreeRingDataloader, labelmeDataset

from INBD.src import (util)

def from_numpy_to_pil(image: np.ndarray) -> Image:
    return Image.fromarray(image)

class TrainingINBD:
    """
    1. Train INBD segmentation model using the given data.
    1.a. Plot validation and training plots
    2. Train INBD network using the given data.
    2.1 hyperparameter tuning
    3. Evaluate the model using the test data
    """
    def __init__(self, dataset_dir, output_dir, inbd_lib_path):
        self.inbd_lib_path = inbd_lib_path
        dataset = labelmeDataset(dataset_dir=dataset_dir, output_dir=output_dir)
        dataset.transform_annotations()
        dataset.split_dataset_in_train_val_and_test()
        train_dataloader, val_dataloader, test_dataloader = dataset.create_dataloaders()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    @staticmethod
    def __run_command(command):
        """
        Run a command
        """
        os.system(command)

        return

    def train_segmentation(self, downsample=4, output_model_dir=None):
        """
        Train the segmentation model
        """
        try:
            model_path = output_model_dir.rglob("*.zip").__next__()
            print("Segmentation Model already exists")

        except StopIteration:
            command = (f"cd {self.inbd_lib_path} && python main.py train segmentation  {self.train_dataloader.images_path} {self.train_dataloader.annotations_path} "
                      # f"--validation_images {self.val_dataloader.images_path} --validation_annotations {self.val_dataloader.annotations_path} "
                       f"--downsample {downsample} --output {output_model_dir}")
            print(command)
            self.__run_command(command)
            model_path = output_model_dir.rglob("*model.pt.zip").__next__()

        return model_path

    def evaluate_segmentation(self, model_path, dataloader,  output_dir=None):
        print(f"Evaluating segmentation model {model_path} on {dataloader.images_path}")
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        segmentationmodel = util.load_segmentationmodel(model_path)
        background_list, center_list, boundary_list, image_list = [], [], [], []
        for idx in tqdm(range(len(dataloader))):
            image, image_path, _, _,  disk_name= dataloader.__getitem__(idx)
            output = segmentationmodel.process_image(str(image_path), upscale_result=False)
            background = output.background
            background -= background.min()
            center = output.center
            center -= center.min()
            boundary = output.boundary
            boundary -= boundary.min()

            #
            background_list.append(from_numpy_to_pil(background).convert("L").resize((640,640)))
            center_list.append(from_numpy_to_pil(center).convert("L").resize((640,640)))
            boundary_list.append(from_numpy_to_pil(boundary).convert("L").resize((640,640)))

            img = from_numpy_to_pil(image)
            # resize image to background shape
            img = img.resize((640,640))

            image_list.append(img)

        images = []
        # generate pdf
        for idx in range(len(background_list)):
            images.append(image_list[idx])
            images.append(boundary_list[idx])
            images.append(background_list[idx])
            images.append(center_list[idx])

        pdf_path = output_dir + "/segmentation.pdf"
        images[0].save(pdf_path, save_all=True, append_images=images[1:], quality=100)
    def train_inbd_network(self, epocs = 1, downsample=4, angular_density=6.28,  output_dir=None, segmentation_model= None):

        command = (f"cd {self.inbd_lib_path} && python main.py train INBD {self.train_dataloader.images_path} {self.train_dataloader.annotations_path} "
                   #f"--validation_images {self.val_dataloader.images_path} --validation_annotations {self.val_dataloader.annotations_path} "
                   f"--downsample {downsample} --output {output_dir} --segmentationmodel={segmentation_model} --per_epoch_it {epocs} --angular-density {angular_density}")
        print(command)
        self.__run_command(command)
        model_path = output_dir.rglob("*model.pt.zip").__next__()
        return model_path

    def evaluate_inbd_network(self, model_path, dataloader,  output_dir=None):
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        pdf_path = output_dir / "inference.pdf"

        print(f"Evaluating inbd model {model_path} on {dataloader.images_path} and saving to {pdf_path}")


        command = (f"cd {self.inbd_lib_path} && python main.py inference {model_path} {dataloader.images_path} "
                   f"--output {output_dir}")
        print(command)
        os.system(command)
        evaluate_output_dir = f"{output_dir}/{output_dir.parent.name}_"
        command = (f"cd {self.inbd_lib_path} && python main.py evaluate {evaluate_output_dir} {dataloader.annotations_path} ")

        print(command)
        os.system(command)

        inference_list, image_list, contour_inference_list = [], [], []
        for idx in tqdm(range(len(dataloader))):
            image, image_path, _, _, disk_name = dataloader.__getitem__(idx)
            imagen_name = Path(image_path).name
            try:

                output_path = Path(output_dir).rglob(f"{imagen_name}*.png").__next__()
                output_contour_path = Path(output_dir).rglob(imagen_name.replace('.jpg',"_debug.png")).__next__()

            except StopIteration:
                #print(output_path)
                #print(output_contour_path)
                continue

            inference = Image.open(output_path)
            contour_inference = Image.open(output_contour_path)
            inference_list.append(inference.convert('RGB').resize((640,640)))
            contour_inference_list.append(contour_inference.convert('RGB').resize((640, 640)))

            img = from_numpy_to_pil(image)
            # resize image to background shape
            img = img.convert('RGB').resize((640, 640))

            image_list.append(img)

        images = []
        # generate pdf
        for idx in range(len(inference_list)):
            images.append(image_list[idx])
            images.append(inference_list[idx])
            images.append(contour_inference_list[idx])


        images[0].save(pdf_path, save_all=True, append_images=images[1:], quality=100)






def train_segmentation( dataset_dir = '/data/maestria/resultados/inbd_pinus_taeda/cstrd/PinusTaedaV1',
                  output_dir = '/data/maestria/resultados/inbd_pinus_taeda/cstrd/inbd',
                  downsample = 4, delete_old_models=False, inbd_lib_path = '/home/henry/Documents/repo/fing/INBD'):

    output_model_dir = Path(output_dir) / f"segmentation_model_downsample_{downsample}"
    if delete_old_models:
        os.system(f"rm -rf {output_model_dir}")
    output_model_dir.mkdir(exist_ok=True, parents=True)

    training = TrainingINBD(dataset_dir, output_dir, inbd_lib_path)
    model_seg = training.train_segmentation(downsample=downsample, output_model_dir=output_model_dir)
    config = dict(model=str(model_seg), output_dir=output_dir, downsample=downsample)
    config_segmentation_path = output_dir + "/config_segmentation.json"
    write_json(config, config_segmentation_path)

    #evaluation
    training.evaluate_segmentation(model_seg, training.val_dataloader, output_dir + "/val")

    training.evaluate_segmentation(model_seg, training.train_dataloader, output_dir + "/train")

    return config_segmentation_path

def train_inbd(dataset_dir = '/data/maestria/resultados/inbd_pinus_taeda/cstrd/PinusTaedaV1',
         output_dir = '/data/maestria/resultados/inbd_pinus_taeda/cstrd/inbd',
         downsample = 4,
         config_segmentation_path = None,
         inbd_lib_path = '/home/henry/Documents/repo/fing/INBD'):

    config_segmentation = load_json(config_segmentation_path)
    model_seg = config_segmentation['model']
    training = TrainingINBD(dataset_dir, output_dir, inbd_lib_path)
    #alpha_values = np.array([1,1.5,2,2.5,3]) * np.pi
    alpha_values = np.array([2]) * np.pi

    epocs_values = [3]
    for alpha in alpha_values:
        for epoc in epocs_values:
                output_dir_inbd = Path(output_dir) / "inbd_training" / f"alpha_{alpha}_epoc_{epoc}_downsample_{downsample}"
                output_dir_inbd.mkdir(exist_ok=True, parents=True)
                existing_models = list(output_dir_inbd.rglob("*model.pt.zip"))
                if not len(existing_models) > 0:
                    model_path = training.train_inbd_network(epocs=epoc, angular_density=alpha, output_dir=output_dir_inbd,
                                                             segmentation_model=model_seg, downsample=downsample)
                else:
                    model_path = existing_models[0]


                output_dir_val = Path(output_dir_inbd) / "val"
                output_dir_val.mkdir(exist_ok=True, parents=True)
                res = training.evaluate_inbd_network(model_path, training.val_dataloader, output_dir_val)

                #output_dir_train = Path(output_dir_inbd) / "train"
                #output_dir_train.mkdir(exist_ok=True, parents=True)
                #res = training.evaluate_inbd_network(model_path, training.train_dataloader, output_dir_train)



    return


def inference_inbd(dataset_dir = '/data/maestria/resultados/inbd_pinus_taeda/cstrd/PinusTaedaV1',
         output_dir = '/data/maestria/resultados/inbd_pinus_taeda/cstrd/inbd',
         model_path = None,
         inbd_lib_path = '/home/henry/Documents/repo/fing/INBD',
         downsample = 4
        ):

    training = TrainingINBD(dataset_dir, output_dir, inbd_lib_path)

    parent_dir = Path(model_path).parent.name
    parent_dir = f"{parent_dir}_downsample_{downsample}"
    output_dir_inbd = Path(output_dir) / "inbd_training" / parent_dir
    output_dir_inbd.mkdir(exist_ok=True, parents=True)

    output_dir_val = Path(output_dir_inbd) / "val"
    output_dir_val.mkdir(exist_ok=True, parents=True)
    res = training.evaluate_inbd_network(model_path, training.val_dataloader, output_dir_val)

    #output_dir_train = Path(output_dir_inbd) / "train"
    #output_dir_train.mkdir(exist_ok=True, parents=True)
    #res = training.evaluate_inbd_network(model_path, training.train_dataloader, output_dir_train)


    output_dir_train = Path(output_dir_inbd) / "test"
    output_dir_train.mkdir(exist_ok=True, parents=True)
    res = training.evaluate_inbd_network(model_path, training.test_dataloader, output_dir_train)


    return

def main(segmentation = False, inbd = False , dataset_dir = '/data/maestria/resultados/inbd_pinus_taeda/cstrd/PinusTaedaV1',
         output_dir = '/data/maestria/resultados/inbd_pinus_taeda/cstrd/inbd',
         downsample = 4,
         delete_old_models=False,
         inbd_lib_path = '/home/henry/Documents/repo/fing/INBD',
         inference = True,
         model_path = None):
    #for downsample in [2, 4]:
    downsample = 1
    if segmentation:
        config_segmentation_path = train_segmentation(dataset_dir, output_dir, downsample, delete_old_models, inbd_lib_path)

    if inbd:
        train_inbd(dataset_dir, output_dir, downsample, config_segmentation_path, inbd_lib_path)

    if inference:
        inference_inbd(dataset_dir, output_dir, model_path, inbd_lib_path)

    return

def parser():
    """
    Inputs
    - segmentation : bool
    - inbd : bool
    - dataset_dir : str
    - output_dir : str
    - downsample : int
    - delete_old_models : bool
    """
    #create the input arguments using argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation", type=bool, default=False)
    parser.add_argument("--inbd", type=bool, default=False)
    parser.add_argument("--dataset_dir", type=str, default='/data/maestria/resultados/inbd_pinus_taeda/cstrd/PinusTaedaV1')
    parser.add_argument("--output_dir", type=str, default='/data/maestria/resultados/inbd_pinus_taeda/cstrd/inbd')
    parser.add_argument("--downsample", type=int, default=3)
    parser.add_argument("--delete_old_models", type=bool, default=False)
    parser.add_argument("--inbd_lib_path", type=str,
                        default='/home/henry/Documents/repo/fing/INBD')
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--inference", type=bool, default=True)
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parser()

    main(args.segmentation, args.inbd,  args.dataset_dir, args.output_dir, args.downsample, args.delete_old_models, args.inbd_lib_path,
         args.inference, args.model_path)

