# Predicting asset ownership in Africa using satellite imagery



## Code structure
* The `scripts` folder contains all the relevant scripts to run our code:
  * `building_dataset.R` unzips and organizes DHS data, merging datasets with their geocoded coordinates, and producing a final dataset with the asset ownership index for each location, and their coordinates. This produces the `data/ground_truth.csv` file.
  * `getting_imagery_mask_gcloud.py` takes the coordinates from each of our villages from `data/ground_truth.csv`, and uses the [Python Google Earth Engine API](https://github.com/google/earthengine-api) to download 1 year composite [Landsat 8 Surface Reflectance](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR) satellite images. These images are stored at `data/imagery/`.
  * `visualizing_imagery.py` grabs the coordinates of a location, and generates a .JPG visualization of the Landsat-8 satellite image for that location.
  * `building_pytorch_imagery.py` grabs the previously downloaded satellite images and converts them to a Pytorch Tensor. We also normalize with the ResNet-18 mean and standard deviations, and split the dataset into training, validation and test sets. This script should produce the `data/pytorch/train_set.pt`, `data/pytorch/val_set.pt` and `data/pytorch/test_set.pt` files.
  * `utils.py` contains various functions we use to train our model or to preprocess data:
    * `preprocess_imagery` convert TIFF images to Numpy arrays, and also corrects images with NaN values.
    * `RegData` is a custom Dataset Class to store a dataset consisting of imagery and labels.
    * `train_model_epoch` trains our model for a full epoch, performing backward and forward propagation.
    * `validation_loop` computes the RMSE of a model in the validation set.
    * `rotation_discrete` is a custom Transform class to apply rotations to images, but only in angles that are multiples of 90 degrees.
    * `train_and_val_model` trains a model to tune its hyperparameter for an arbitrary number of epochs. It calls `train_model_epoch` and `validation_loop`.
    * `model_frozen` defines a ResNet-18 model with all the conv2d layers frozen, except for the last "n_unfreeze" layers.
    * `experiment_store` reads a dictionary with the data from our experiments, and stores that as CSV files.
    * `test_set_rmse_and_save` is used at the end of our experimentation to compute the RMSE of our best model in the best set, and to store the predictions for the test set.
  * `training_models.py` runs experiments on different Pytorch models to helps us decide how to tune the hyperparameters in our model.
  * `PENDING.R` produces a heatmap of the average asset ownership per country in Africa, first using the labels in our test set, and then using the predictions of our best model for those labels.
