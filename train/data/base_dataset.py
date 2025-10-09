from torch.utils.data import Dataset
from .download_utils import download_dataset
import os
#The single dot (.) indicates that download_utils is a module located in the 
#same package as the current module where this import statement is written.

class BaseDataset(Dataset):
    def __init__(self, root, filename = None, download_url=None, force_download = False):
        self.root_path = root
        if download_url is not None: #if url is given
            if filename == None:
              dataset_name = download_url[download_url.rfind('/')+1:]
              if dataset_name.endswith('.zip'):
                dataset_name = dataset_name[:-4]
            else:
              dataset_name = filename
            #'rfind'searches for the index of '/' from right to left, unlike the 
            #'find' method, which searches from left to right.
            #Using slicing ':' in index we take substring from found index until
            #the end
            download_dataset(url=download_url, root = root, filename = dataset_name, force_download = force_download)
            self.dataset_name = dataset_name
            
