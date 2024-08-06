# ARGfore: A multivariate time-series forecasting framework for predicting abundance of antibiotic resistance genes

The global spread of antibiotic resistance genes (ARGs) presents a significant health threat to humans, animals, and plants, thus calling for effective monitoring and mitigation strategies. Metagenomic sequencing is increasingly being utilized to profile ARGs in various environments, but presently a mechanism for predicting future trends in ARG occurrence patterns is lacking. The ability to forecast ARG abundance trends would be extremely valuable to inform proactive control and mitigation strategies. Such a tool could be applied to a variety of environments and purposes of interest, including wastewater-based surveillance.

Here we propose ARGfore, a multivariate time-series forecasting model for the prediction of ARG abundance from time-series metagenomic data using  deep neural networks. ARGfore extracts features that capture the inherent relationships among ARGs within the same drug classes and is trained to recognize patterns in ARG trends and seasonality. ARGfore outperformed standard  time-series forecasting methods, exhibiting the lowest mean absolute percentage error when applied to wastewater datasets. Additionally, ARGfore demonstrated enhanced computational efficiency in ARG abundance prediction, making it a promising candidate for a variety of surveillance applications. The rapid prediction of future trend can facilitate early detection and deployment of mitigation efforts if necessary.

## Requirements
* Python (>= 3.6)
* Tensorflow (>= v1.8.0)
* Other python packages : numpy, pandas, os, sys, scikit-learn

## Usage
Clone the repository or download source code files.

## Inputs
[Note!] All the example datasets can be found in './example/' directory.

#### 1. Time-series ARG abundance files
* Contains ARG abundance profiles for each timepoint
* Row : Timepoint (Sample), Column : Feature (ARG)
* The first colum name should be the "Time" having timepoint information, where each timepoint needs to be denoted as the format of "m/d/y" (e.g., 8/10/20).
* The first row should contain the ARG names.
* Example : ./example/example_arg_dataset.csv

#### 2. Drug information of ARGs
* Contains drug class information of ARGs in time-series ARG abundance data
* Row : ARG, Column : Drug class
* The first colum name should be the "gene" having ARG names, and the second column name should be the "drug".
* Example : ./example/example_drug_info.csv

#### Parameters
* H : the length of timepoints for the forecast period that the model will predict (e.g., 10)
* n : Factor to be multipled to H (Determine the lengh of input n\*H (e.g., 5\*10))

## How to run
1. Edit the **run_ARGfore.sh** to make sure each variable indicate the corresponding files and paramter values as input.
2. Run the below command :
```
chmod +x run_ARGfore.sh
./run_ARGfore.sh
```
(Note! Current ARGfore file splits the dataset to 8:2 ratio for training and testing, which reproduces the same experimental setting in the paper)

3. All the results will be saved in the newly created **results** directory.
   * final_inverse_result.csv : forecasted ARG abundance values in the original data scale
   * final_result.csv : forecasted ARG abundance values in the range of 0-1 (min-max normalized values)

## Contact
If you have any questions or problems, please contact to **joungmin AT vt.edu**.
