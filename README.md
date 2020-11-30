# uvgg19_crack_detection
Code used to produce the results presented in _insert our reference_ 

This repository is compatible with CrackForest [https://github.com/cuilimeng/CrackForest-dataset] dataset; Aigle-RN and ESAR (2 out of the 3 parts of the "CrackDataset")[https://www.irit.fr/~Sylvie.Chambon/Crack_Detection_Database.html] datasets; and the cropped GAPs384 and cracktree200 [https://github.com/fyangneil/pavement-crack-detection] datasets.

_You can download the datasets from the corresponding links, or use our links below. In any case, don't forget to cite the sources._

## Pre-requisites
This repository was tested on two setups:
* Ubuntu 18.04 with a Nvidia GeForce GTX 1050 using Driver Version 440.82 and CUDA Version 10.2
* Ubuntu 20.04 with a Nvidia GeForce RTX 2070 using Driver Version 450.80.02 and CUDA Version 11.0

The network was build using Tensorflow 2.1.0. An environment.yml is provided in this repository to clone the environment used (recommended).

## How to run
### Train and validate
You can train and validate a model on a single dataset. For example, to train and validate on CrackForest with default parameters, run:
```
python train_and_validate.py --dataset_names "cfd" --dataset_paths "path/to/cfd"
```

You can create a bigger dataset by combining 2 or more datasets. For example, to train and validate on CrackForest and Aigle-RN combined using default parameters, run:
```
python train_and_validate.py --dataset_names "cfd" "aigle-rn" --dataset_paths "path/to/cfd" "path/to/crackdataset"
```

The program will then train the default model using the listed datasets. Training images are split into training and validation images using a 80/20 proportion; the model will be trained until the desired number of epochs (default: 150) or if the loss in the validation images doesn't improve for 20 epochs.

After the training is done, a results_date_time folder will be created. It contains a csv file with the training history and a plot of such training history. Additionally, there are 4 folders:
* "results_training": it contains a visual comparison of ground truth and predicted cracks in the images used for training. It contains a hdf5 with the weights from the last training epoch too.
* "results_training_min_val_loss": the same as before, but using the weights from the epoch with the minimum loss in the test images during training (only when the model is trained more than 1 epoch).
* "results_test": it contains a visual comparison of ground truth and predicted cracks in the images used for testing. It contains a text file with numeric metrics of the performance in the test set.
* "results_test_min_val_loss": the same as before, but using the weights from the epoch with the minimum loss in the test images during training (only when the model is trained more than 1 epoch).

You can "train" the model using 0 epochs. In this case, the results will be obtained using the initial weights of the network, or pre-trained weights if they are provided as argument.

#### Input arguments

* ("--dataset_names", type=str, nargs="+", help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar', 'crack500', 'gaps384', 'cracktree200'")
* ("--dataset_paths", type=str, nargs="+", help="Path to the folders containing the respective datasets as downloaded from the original source.")
* ("--model", type=str, default="uvgg19", help="Network to use. It can be either a name from 'models.available_models.py' or a path to a json file.")
* ("--training_crop_size", type=int, nargs=2, default=[256, 256], help="For memory efficiency and being able to admit multiple size images, subimages are created by cropping original images to this size windows")
* ("--alpha", type=float, default=3.0, help="Alpha for objective function: BCE_loss + alpha*DICE")
* ("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
* ("--epochs", type=int, default=150, help="Number of epochs to train.")
* ("--batch_size", type=int, default=4, help="Batch size for training.")
* ("--pretrained_weights", type=str, default=None, help="Load previous weights from this location.")
* ("--use_da", type=str, default="False", help="If 'True', training will be done using data augmentation. If 'False', just raw images will be used.")
* ("--save_validation_paths", type=str, default="False", help="If 'True', a text file 'validation_paths.txt' containing the paths of the images used for validating will be saved in the project's root.")

### Validate a trained model
For cross-dataset validation, you can validate a pre-trained model on a full dataset (or a set of datasets) by running:
```
python validate.py dataset_name(or names separated by space) path/to/dataset(or paths separated by space) model(either "uvgg19" to use our architecture or a path to a json model) path/to/pretrained_weights
```

Notice that for both, train_and_validate.py and validate.py, you can provide specific image paths as a dataset. To do this, provide "text" as dataset name; as path/to/dataset, use a path to a text file containing the desired image paths (use --save_validation_paths True in train_and_validate.py to see an example of the needed format).

### Compare overlaid ground-truth and prediction
To get a color-coded comparison like in Figure 2 from the paper, run:
```
python -c "from data import analyse_resulting_image_folder. analyse_resulting_image_folder(path/to/results_folder)"
```
results_folder can be any of the 4 folders created inside the output folder of train_and_validate.py (e.g. "results_test_min_val_loss").