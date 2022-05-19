# PINPOINT: Privacy-preserving time series prediction with temporal convolutional neural networks

This repository contains the code and experiments used in the paper "Privacy-preserving time series prediction with
temporal convolutional neural networks", to appear in the proceedings of IJCNN 2022.

## Organization
- `data`: folder containing the time-series data used in the experiments;
- `EncryptedExperiments`: folder containing the notebooks which shows how a PINPOINT model, trained on plain data, can be used to make forecasts on encrypted data. The notebooks called "x.nbconvert.ipynb" are simply the run result of the corresponding notebook "x.ipynb";
- `Experiments`: folder containing the notebooks used to train PINPOINT models and compute the error metrics. After running each notebook, the PINPOINT models are saved in the `models` folder, while a csv file containing the metrics is saved in `results`. Finally, the `CreateFinalCSV.ipynb` notebooks output the `final_results.csv` file, which basically contains the table showed in the paper;
- `pycrcnn`: this folder contains the code needed to execute PINPOINT models on encrypted data. In particular, the code is taken from the ![PyCrCNN](https://github.com/AlexMV12/PyCrCNN), a library for HE-enabled Convolutional Neural Networks developed by us, and published at ![IJCNN 2019](https://ieeexplore.ieee.org/abstract/document/9207619/);
- `requirements.txt` contains the requirement needed to execute all the code. In particular, the ![Pyfhel 2.3.1](https://github.com/ibarrond/Pyfhel/releases/tag/v2.3.1) library should be installed;
- `train_utils.py` contains some code used in the notebooks.

## Experimental overview
Consider the experiments for the ![Milk consumption](https://rdrr.io/cran/fma/man/milk.html) dataset:
1. Run the Experiments/Milk.ipynb notebook. It will create PINPOINT-1CONV, PINPOINT-2CONV, Prophet, ARIMA, ARIMA-PP, and Naive models, train them, and show the result on the testing set;
2. Then, run the EncryptedExperiments/PINPOINT-1CONV/Milk.ipynb notebook. It will load the PINPOINT-1CONV model produced in the previous step (in particular, the one with highest forecast horizon which is basically the "heaviest"). Then, it will run it on the *encrypted* test set of the Milk dataset. It will show that the results obtained are the same obtained on plain data.
3. The same steps hold for all the the datasets. Times and memory notebooks/scripts are present in the same folders.

Consider also that, in general, encryption parameters should be tuned for each use-case. The ones presented in the notebooks are a suggestion; if one wants an higher precision (i.e., a lower error introduced in the encrypted processing) the value of `p` should be increased; if the Noise Budget runs out, also `m` may be increased. The reader may refer the related literature for more info. Also, the notebook `EncryptedExperiments/NoiseBudget_Tester.ipynb` may be useful to test different encryption parameters.

