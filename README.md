# Predicting asset ownership in Africa using satellite imagery



## Code structure
* The `scripts` folder contains all the relevant scripts to run our code:
  * `building_dataset.R` unzips and organizes DHS data, merging datasets with their geocoded coordinates, and producing a final dataset with the asset ownership index for each location, and their coordinates. This produces the `data/ground_truth.csv` file.
  * `getting_imagery_mask_gcloud.py` takes the coordinates from each of our villages from `data/ground_truth.csv`, and uses the [Python Google Earth Engine API](https://github.com/google/earthengine-api) to download 1 year composite [Landsat 8 Surface Reflectance](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR) satellite images. These images are stored at `data/imagery/`.
  * 
