# uvgg19_crack_detection
Code used to produce the results presented in:
```
@article{neurocomputing,
    title = {Pixel-accurate road crack detection in presence of inaccurate annotations},
    journal = {Neurocomputing},
    volume = {480},
    pages = {1-13},
    year = {2022},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2022.01.051},
    url = {https://www.sciencedirect.com/science/article/pii/S0925231222000728},
    author = {Rodrigo Rill-García and Eva Dokladalova and Petr Dokládal},
}    
```
If using this code, kindly cite the paper above.

This repository is compatible with CrackForest [https://github.com/cuilimeng/CrackForest-dataset] dataset; Aigle-RN and ESAR (2 out of the 3 parts of the "CrackDataset")[https://www.irit.fr/~Sylvie.Chambon/Crack_Detection_Database.html] datasets; the cropped GAPs384 and cracktree200 [https://github.com/fyangneil/pavement-crack-detection] datasets; and Syncrack generated [https://github.com/Sutadasuto/syncrack_generator] datasets.

_You can download the datasets from the corresponding links, or use our link below (recommended). In any case, don't forget to cite the sources._

https://drive.google.com/drive/folders/1g-nQchtC7Adjk87KIx2Ber-10EbpHGKB?usp=sharing

We provide the pruned version of CFD as described in section 4.2. "CrackDataset" contains both Aigle-RN and ESAR; when using any of these 2 datasets, the path to "CrackDataset" should be provided.

In the next link, we provide the weights from the model trained on each one of the provided datasets:

https://drive.google.com/drive/folders/1saPBK_0pfn-dIYF7OaBu5mukF5o4jgME?usp=sharing

## Pre-requisites
This repository was tested on two setups:
* Ubuntu 18.04 with a Nvidia GeForce GTX 1050 using Driver Version 440.82 and CUDA Version 10.2
* Ubuntu 20.04 with a Nvidia GeForce RTX 2070 using Driver Version 450.80.02 and CUDA Version 11.0

The network was build using Tensorflow 2.1.0. An environment.yml is provided in this repository to clone the environment used (recommended).

## How to run
### Training and validating a model
You can train and validate a model on a single dataset. For example, to train and validate on CrackForest with default parameters, run:
```
python train_and_validate.py -d "cfd" -p "path/to/cfd"
```

You can create a bigger dataset by combining 2 or more datasets. For example, to train and validate on CrackForest and Aigle-RN combined using default parameters, run:
```
python train_and_validate.py -d "cfd" "aigle-rn" -d "path/to/cfd" "path/to/crackdataset"
```

The program will then train the default model using the listed datasets. Available images are split into training and validation images using a 80/20 proportion; for reproducibility, a fixed random seed is used. The model is trained until reaching the desired number of epochs (default: 150) or if the loss in the validation images doesn't improve for 20 epochs.

After the training is done, a results_date_time folder will be created. It contains a csv file with the training history and a plot of such training history. Additionally, there are 4 sub-folders:
* "results_training": it contains a visual comparison of ground truth and predicted cracks in the images used for training. It contains a hdf5 file with the weights from the last training epoch too.
* "results_training_min_val_loss": the same as before, but using the weights from the epoch with the minimum loss in the validation images during training (only when the model is trained more than 1 epoch).
* "results_test": it contains a visual comparison of ground truth and predicted cracks in the images used for validation. It contains a text file with numeric metrics of the performance in the validation set.
* "results_test_min_val_loss": the same as before, but using the weights from the epoch with the minimum loss in the validation images during training (only when the model is trained more than 1 epoch).

You can "_train_" the model using 0 epochs. In this case, the results will be obtained using the initial weights of the network, or pre-trained weights if they are provided as argument.

#### Input arguments

* ("-d", "--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'aigle-rn', 'esar', 'gaps384', 'cracktree200', 'text'")
* ("-p", "--dataset_paths", type=str, nargs="+",
                    help="Path to the folders or files containing the respective datasets as downloaded from the original source.")
* ("-m", "--model", type=str, default="uvgg19",
                    help="Network to use. It can be either a name from 'models.available_models.py' or a path to a json file.")
* ("-w", "--pretrained_weights", type=str, default=None,
                    help="Load previous weights from this location.")
* ("-da", "--use_da", type=str, default="False", help="If 'True', training will be done using "
                                                                       "data augmentation. If 'False', just raw "
                                                                       "images will be used.")
* ("--training_crop_size", type=int, nargs=2, default=[256, 256],
                    help="For memory efficiency and being able to admit multiple size images,subimages are created by cropping original images to this size windows")
* ("--alpha", type=float, default=3.0,
                    help="Alpha for objective function: BCE_loss + alpha*DICE")
* ("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
* ("--epochs", type=int, default=150, help="Number of epochs to train.")
* ("--batch_size", type=int, default=4, help="Batch size for training.")
* ("--patience", type=int, default=20, help="Early stop patience.")
* ("--save_validation_paths", type=str, default="False", help="If 'True', a text file "
                                                                               "'validation_paths.txt' containing "
                                                                               "the paths of the images used "
                                                                               "for validating will be saved in "
                                                                               "the project's root.")
### Validating a trained model with a different dataset
For cross-dataset validation, you can validate a pre-trained model on a full dataset (or a set of datasets) by running:
```
python validate.py -d dataset_name(or names separated by space) -p path/to/dataset(or paths separated by space) -w path/to/pretrained_weights
```

Notice that for both, _train_and_validate.py_ and _validate.py_, you can provide specific image paths as a dataset. To do this, provide "_text_" as dataset name; as _path/to/dataset_, use a path to a text file containing the desired image paths (use _--save_validation_paths True_ in _train_and_validate.py_ to see an example of the required format).

### Comparing ground-truth and prediction from a validation script
To get a color-coded comparison like in Figure 2 from the paper, run:
```
python -c "from data import analyse_resulting_image_folder; analyse_resulting_image_folder('path/to/results_folder')"
```
_results_folder_ can be any of the 4 folders created inside the output folder created by _train_and_validate.py_ (e.g. "results_test_min_val_loss").

### Using a trained model to predict a single image or a dataset
You can use a trained model to predict a single image by running:
```
python predict_image.py path/to/image path/to/pretrained_weights
```
The resulting image will be saved in the project's root.

You can also predict a whole dataset (without any kind of evaluation) by running:
```
python predict_dataset.py -d dataset_name(or names separated by space) -p path/to/dataset(or paths separated by space) -w path/to/pretrained_weights
```
The resulting images will be saved inside a folder "results" in the project's root.

### Analyzing intensities of pixels inside the ground-truth and predicted masks
Once you have a trained model, you can analyze its predictions from a given dataset in terms of pixel-intensities (see paper, section V.B). A comparison is done with respect to the given ground-truth segmentation:
```
python validate_model_intensities_on_dataset.py -d dataset_name(or names separated by space) -p path/to/dataset(or paths separated by space) -w path/to/pretrained_weights
```
A file "intensity_validation.csv" will be created in the project's root. This shows (per input image) the average intensity and the standard deviation of the pixels in the ground-truth mask and in the predicted mask. In other words, it ignores any pixel in the input image that is not considered as crack according to the segmentation masks.

At the end of the file, the average of the previous scores is also presented.
