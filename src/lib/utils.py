import json

def polygon_2_labelme_json(chain_list, image_height, image_width,  image_path):
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

    labelme_json = {"imagePath":str(image_path), "imageHeight":None,
                    "imageWidth":None, "version":"5.0.1",
                    "flags":{},"shapes":[],"imageData": None}
    for idx, polygon in enumerate(chain_list):
        if len(polygon.shape) < 2 :
            continue

        ring = {"label":str(idx+1)}

        ring["points"] = polygon.tolist()
        ring["shape_type"] = "polygon"
        ring["flags"] = {}
        labelme_json["shapes"].append(ring)

    return labelme_json
