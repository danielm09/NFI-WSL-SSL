# Download data
Run download_data.py to download train and test datasets for the weakly supervised learning (WSL) and train set for the masked autoencoder (MAE).

# Usage
Download all datasets

```
python download_data.py
```

Or choose specific datasets to download
```
python download_data.py train_wsl test_wsl
```
> *Options: train_wsl, test_wsl, train_mae*
>
> Make sure download_data.py is called from the 'data' folder

# Alternative
Download the data directly from AWS S3.


[https://ifn-wsl-ssl-data.s3.eu-west-3.amazonaws.com/crops_train_seg_all_64x64_181b_augmented.hdf5](https://ifn-wsl-ssl-data.s3.eu-west-3.amazonaws.com/crops_train_seg_all_64x64_181b_augmented.hdf5)


[https://ifn-wsl-ssl-data.s3.eu-west-3.amazonaws.com/crops_test_seg_all_64x64_181b.hdf5](https://ifn-wsl-ssl-data.s3.eu-west-3.amazonaws.com/crops_test_seg_all_64x64_181b.hdf5)


[https://ifn-wsl-ssl-data.s3.eu-west-3.amazonaws.com/crops_train_all_SSL.hdf5](https://ifn-wsl-ssl-data.s3.eu-west-3.amazonaws.com/crops_train_all_SSL.hdf5)
