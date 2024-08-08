# MLBrief: INBD


##


## Transform INBD prediction to UruDendro format
In order to transform the INBD predictions to the labelme format, you can use the following command:
```bash
python src/from_inbd_to_urudendro_labels.py --root_dataset DATASET_PATH --root_inbd_results INBD_RESULTS_PATH --output OUTPUT_PATH
```

Where:
- `DATASET_PATH` is the path to the dataset folder.
- `INBD_RESULTS_PATH` is the path to the INBD results folder.
- `OUTPUT_PATH` is the path to the output folder.

## Transform UruDendro annotations to INBD format
In order to transform the UruDendro annotations to the INBD format, you can use the following command:
```bash
python src/dataset_urudendro.py --dataset_dir DATASET_PATH --output_folder OUTPUT_PATH --size SIZE
```

Where:
- `DATASET_PATH` is the path to the dataset folder.
- `OUTPUT_PATH` is the path to the output folder.
- `SIZE` is the size of the images.

## Resize INBD dataset
In order to resize the INBD dataset, you can use the following command:
```bash
python src/dataset_inbd.py --dataset_dir DATASET_PATH --output_folder OUTPUT_PATH --size SIZE
```

Where:
- `DATASET_PATH` is the path to the dataset folder.
- `OUTPUT_PATH` is the path to the output folder.
- `SIZE` is the size of the images.


