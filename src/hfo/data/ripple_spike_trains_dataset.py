from torch.utils.data import Dataset
import numpy as np
from utils.io import preview_np_array
from torch import from_numpy

'''
Transforms the spike train data into a format that can be fed to the network.
@up_spike_train (array): Spike train data for the UP direction
@down_spike_train (array): Spike train data for the DOWN direction

Returns:
- up_spike_times (array): Spike times for the UP direction
- down_spike_times (array): Spike times for the DOWN direction
'''
def prepare_spikes_data(up_spike_train, down_spike_train):
    # To feed the spike trains to the network, the channel_idx is not necessary. Thus, let's remove the extra dimension
    up_spike_times = up_spike_train[:, 0]
    down_spike_times = down_spike_train[:, 0]
    
    # To combine the spike trains UP and DOWN, they both need to have the same dimensions.
    # Let's take the minimum length of the two spike trains and crop the longer one.
    len_smaller_spike_train = min(up_spike_times.shape[0], down_spike_times.shape[0])
    up_spike_times = up_spike_times[:len_smaller_spike_train]
    down_spike_times = down_spike_times[:len_smaller_spike_train]

    return up_spike_times, down_spike_times

class SpikeTrainsDataset(Dataset):
    '''
    Dataset class for spike trains data.
    '''
    
    def __init__(self, spikes_trains_filename, annotations_filename, transform=None, target_transform=None, verbose=False):
        '''
        Constructor of the Dataset.
        @spike_trains_filename (str): Name of the file containing the SEEG data converted to UP and DOWN spikes.
        @down_filename (str): Name of the file containing the DOWN spike train data.
        @annotations_filename (str): Name of the file containing the annotations.
        @transform (callable, optional): Optional transform to be applied on the data
        @target_transform (callable, optional): Optional transform to be applied on the labels
        '''
        self.input_spikes = np.load(f"{spikes_trains_filename}")
        if verbose:
            preview_np_array(self.input_spikes, "Input Spikes")
            
        # Convert the input_spikes to a Tensor
        self.input_spikes = from_numpy(self.input_spikes)
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        '''
        Returns the number of samples in the dataset.
        '''
        return self.input_spikes.shape[0]
    
    def __getitem__(self, idx):
        '''
        Returns the sample at the given index.
        @idx (int): Index of the sample to retrieve.

        Returns:
        - spike_array (array): spike train data (binary) for the given index (UP and DOWN)
        '''
        # Get the spike train data for the given index
        curr_spike_train = self.input_spikes[idx]

        # Get the labels for the given index
        # TODO
        label = False

        # Apply the transformation to the input
        if self.transform:
            curr_spike_train = self.transform(curr_spike_train)

        # Apply the transformation to the labels
        if self.target_transform:
            # TODO
            pass

        # Return the spike train data
        return curr_spike_train, label
    