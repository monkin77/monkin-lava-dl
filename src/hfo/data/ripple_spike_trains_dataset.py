import os
import pandas as pd
from torch.utils.data import Dataset
from utils.input import read_spike_events

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
    
    '''
    Constructor of the Dataset.
    @up_filename (str): Name of the file containing the UP spike train data.
    @down_filename (str): Name of the file containing the DOWN spike train data.
    @annotations_filename (str): Name of the file containing the annotations.
    @transform (callable, optional): Optional transform to be applied on the data
    @target_transform (callable, optional): Optional transform to be applied on the labels
    '''
    def __init__(self, up_filename, down_filename, annotations_filename, transform=None, target_transform=None):
        ripple_spike_train_up = read_spike_events(up_filename)
        ripple_spike_train_down = read_spike_events(down_filename)
        self.up_spike_times, self.down_spike_times = prepare_spikes_data(ripple_spike_train_up, ripple_spike_train_down)
        print(f"UP spike train data: {self.up_spike_times.shape}")
        print(f"DOWN spike train data: {self.down_spike_times.shape}")
        
        self.transform = transform
        self.target_transform = target_transform
    
    '''
    Returns the number of samples in the dataset.
    '''
    def __len__(self):
        
    