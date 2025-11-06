from torch.utils.data import Dataset
from .compile_utils import compile_dataset
import os
#The single dot (.) indicates that download_utils is a module located in the 
#same package as the current module where this import statement is written.

class BaseDataset(Dataset):
		def __init__(self, root="compiled_data", filename = "data.npz", participant_path=None, force_compile = False, processing_params=None):
				self.root_path = root
				self.processing_params = processing_params
				#TODO: instead of downlod_url, user could pass the path to the participant Data. If force=True, then recompile the data.
				if participant_path is not None: #if url is given
						compile_dataset(path=participant_path, root = root, filename = filename, processing_params=processing_params, force_compile = force_compile)
						self.dataset_name = filename
