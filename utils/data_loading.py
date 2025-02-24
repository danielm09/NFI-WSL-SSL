import torch
from torch.utils.data import Dataset
import numpy as np
import h5py



class WSL_Dataset(Dataset):
    def __init__(self, hdf5_path, bands_idxs=None, transform=None, standardize=False, means_np=None, stds_np=None, downsample_classes=None):
        """
        Defines a Dataset object. It will read data from a hdf5 file.

        Args:
            hdf5_path (str): string containing the path pointing to hdf5 file (file should have keys 'crops' and 'labels');
            bands_idxs (list): array with indices of bands, if one wants to use a subset of the bands;
            transform: transforms to be applied to the data;
            standardize (boolean): whether the data must be standardized (if True, one should provide means_np and stds_np);
            means_np (ndarray): numpy array with pre-calculated means of each band (order should match crops bands order);
            stds_np (ndarray): numpy array with pre-calculated stds of each band (order should match crops bands order);
            downsample_classes (list): array with labels of classes to be downsampled.
        """
        self.hdf5_path = hdf5_path
        self.h5file = h5py.File(hdf5_path, 'r') #"crops" (n, b, w, h), "labels" (n, w, h)
        self.size = self.h5file['labels'].shape[0] #number of examples
        if bands_idxs:
            self.bands_idxs = bands_idxs
        else: #if no bands_idxs are provided, then use all bands
            self.bands_idxs = [i for i in range(self.h5file['crops'].shape[1])]
        self.transform = transform
        self.standardize = standardize
        if standardize:
            if means_np is None or stds_np is None:
                raise TypeError("When standardize is True, provide means_np and stds_np")
            if type(means_np)!=np.ndarray or type(stds_np)!=np.ndarray:
                raise TypeError("means_np and stds_np should be numpy arrays")
            self.means = torch.from_numpy(means_np.astype(np.float32)).view(means_np.shape[0], 1, 1)
            self.stds = torch.from_numpy(stds_np.astype(np.float32)).view(stds_np.shape[0], 1, 1)
        self.downsample_classes = downsample_classes

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if self.standardize:
            crop = torch.from_numpy(self.h5file['crops'][idx, self.bands_idxs, :, :].astype(np.float32))
            crop = torch.where(crop > 10000, 10000, crop) #normalization (max Sentinel-2 value capped at 10000)
            crop = (crop - self.means)/self.stds #standardization
        else:
            crop = torch.from_numpy(self.h5file['crops'][idx, self.bands_idxs, :, :].astype(np.float32) / 10000.0)
            crop = torch.where(crop > 1, 1.0, crop) #normalization (max Sentinel-2 value capped at 10000)

        label = torch.as_tensor(self.h5file['labels'][idx],dtype=torch.long)
        
        #downsample selected classes (when retrieving a crop, randomly mask a number of pixels)
        if self.downsample_classes:
            if label.min() in self.downsample_classes:
                label = self.keep_two_random_elements(label)

        label[label==255]=20 #sets the unlabeled pixels (255) to 20 (extra class)

        if self.transform:
            # Concatenate the label as an additional channel to the crop
            combined = torch.cat((crop, label.unsqueeze(0)), dim=0)
            combined = self.transform(combined)

            # Split the crop and label back into separate tensors
            crop = combined[:-1]  # All but the last channel
            label = combined[-1].long()  # The last channel

        return crop, label

    def keep_two_random_elements(self, tensor):
        # Get indices where the values are not 255 (valid values)
        valid_indices = (tensor != 255).nonzero(as_tuple=False)
        
        # If there are more than 2 valid values, randomly sample 3 indices
        if valid_indices.size(0) > 3:
            selected_indices = valid_indices[torch.randperm(valid_indices.size(0))[:3]]
            
            # Create a mask for all valid values, set them to 255
            mask = torch.ones_like(tensor, dtype=torch.bool)
            mask[selected_indices[:, 0], selected_indices[:, 1]] = False
            
            # Set all elements except the selected ones to 255
            tensor[mask] = 255

        return tensor
        

class SSL_Dataset(Dataset):
    def __init__(self, hdf5_path, bands_idxs, transform=None, standardize=False, means_np=None, stds_np=None):
        """
        Defines a Dataset object. It will read data from a hdf5 file.

        Args:
            hdf5_path (str): string containing the path pointing to hdf5 file (file should have keys 'crops');
            bands_idxs (list): array with indices of bands, if one wants to use a subset of the bands;
            transform: transforms to be applied to the data;
            standardize (boolean): whether the data must be standardized (if True, one should provide means_np and stds_np);
            means_np (ndarray): numpy array with pre-calculated means of each band (order should match crops bands order);
            stds_np (ndarray): numpy array with pre-calculated stds of each band (order should match crops bands order).
        """

        self.hdf5_path = hdf5_path
        self.h5file = h5py.File(hdf5_path, 'r') #"crops" (n, b, w, h)
        self.size = self.h5file['crops'].shape[0] #number of examples
        if bands_idxs:
            self.bands_idxs = bands_idxs
        else: #if no bands_idxs are provided, then use all bands
            self.bands_idxs = [i for i in range(self.h5file['crops'].shape[1])]

        self.transform = transform
        self.standardize = standardize
        if standardize:
            if means_np is None or stds_np is None:
                raise TypeError("When standardize is True, provide means_np and stds_np")
            if type(means_np)!=np.ndarray or type(stds_np)!=np.ndarray:
                raise TypeError("means_np and stds_np should be numpy arrays")
            self.means = torch.from_numpy(means_np.astype(np.float32)).view(means_np.shape[0], 1, 1)
            self.stds = torch.from_numpy(stds_np.astype(np.float32)).view(stds_np.shape[0], 1, 1)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        crop = torch.from_numpy(self.h5file['crops'][idx].astype(np.float32))
        crop = torch.where(crop > 10000, 10000, crop) #normalization
        crop = (crop - self.means)/self.stds #standardization

        if self.transform:
            crop = self.transform(crop)
        return crop
    
