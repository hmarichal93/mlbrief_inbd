import numpy as np
import argparse
import cv2
from PIL import Image
from shapely.geometry import Polygon
from pathlib import Path

def generate_mask_image_from_txt(filename, img_path, output_img):
    import os
    #open txt with pandas
    f = open(filename, "r")
    poly = []
    for line in f:
        print(line)
        y, x = line.replace("[", "").replace("]", "").split(",")
        poly.append([float(y), float(x)])

    f.close()

    #print(df)
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, np.array([poly], dtype=np.int32), 255)
    #save mask
    cv2.imwrite(output_img, mask)
    #print(mask_path)
    #os.system("ls -lah .")
    #raise
    return

def resize_image_using_pil_lib(im_in: np.array, height_output: int, width_output: int, keep_ratio= True) -> np.ndarray:
    """
    Resize image using PIL library.
    @param im_in: input image
    @param height_output: output image height_output
    @param width_output: output image width_output
    @return: matrix with the resized image
    """

    pil_img = Image.fromarray(im_in)
    # Image.ANTIALIAS is deprecated, PIL recommends using Reampling.LANCZOS
    flag = Image.ANTIALIAS
    # flag = Image.Resampling.LANCZOS
    if keep_ratio:
        aspect_ratio = pil_img.height / pil_img.width
        if pil_img.width > pil_img.height:
            height_output = int(width_output * aspect_ratio)
        else:
            width_output = int(height_output / aspect_ratio)

    pil_img = pil_img.resize((width_output, height_output), flag)
    im_r = np.array(pil_img)
    return im_r

def resize_image(img_path, mask_path, hsize, wsize):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if hsize == 0 or wsize == 0:
        return

    img_r = resize_image_using_pil_lib(img, hsize, wsize)
    mask_r = resize_image_using_pil_lib(mask, hsize, wsize)


    import os
    os.system("rm -f " + img_path)
    os.system("rm -f " + mask_path)

    cv2.imwrite(img_path, img_r)
    cv2.imwrite(mask_path, mask_r)

    return


def main(filename, img_path, mask_path, hsize = None, wsize = None):

    generate_mask_image_from_txt(filename, img_path, mask_path)

    resize_image(img_path, mask_path, hsize, wsize)

    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_poly", type=str, required=True)
    parser.add_argument("--input_img", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--hsize", type=int, required=False, default=0)
    parser.add_argument("--wsize", type=int, required=False, default=0)
    args = parser.parse_args()

    main(args.input_poly, args.input_img, args.mask_path, args.hsize, args.wsize)



