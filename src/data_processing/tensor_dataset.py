import os
import pandas as pd

from torch.utils.data import Dataset
from src.data_processing.data_utils import get_col_info, organize_labels, df_to_tensor, get_col_dims, make_binary
from src.data_processing.tackle_missing_value import tackle_missing_value_wrapper

CURDIR = os.path.dirname(__file__)

class TEDSTensorDataset(Dataset):
    """
    PyTorch Dataset for the TEDS (Temporal Embedding Deep Sequence) model.

    This dataset converts and stores the original data in the form of **pure PyTorch tensors**
    instead of `pyg.Data` or `pyg.Batch`, in order to avoid storage inefficiency and unnecessary
    object-creation overhead. The data include only two temporal snapshots: admission and
    discharge.

    Attributes:
        root (str): Root directory for storing and loading the dataset.
        processed_tensor (torch.Tensor): Final preprocessed data tensor (inputs + labels).
            Shape: (num_samples, num_features).
            This tensor is later split into X and y when passed to the DataLoader.
        col_info (tuple[list[int], list[int]]): Column index information.
            (List of column indices at admission, list of column indices at discharge)
        LOS (pandas.Series): Length of Stay (LOS) information.
    """
    def __init__(self, root:str, binary=True, ig_label=False):
        """
        Constructor for the TEDSTensorDataset.

        This initializes dataset paths, creates required directories, and either loads
        previously processed data or performs a new preprocessing step and loads the
        resulting data into memory.

        Args:
            root (str): Root directory path where the dataset is stored.
            binary (bool): Whether to convert REASON into its binary form (REASONb).
            ig_label (bool): Whether to newly include Neutral labels for Integrated Gradients
        """
        super().__init__()
        self.binary = binary
        self.ig_label = ig_label

        self.root = root
        self.raw_data_path = os.path.join(self.root, 'raw', 'TEDS_Discharge.csv')
        self.missing_corrected_path = os.path.join(self.root, 'raw', 'missing_corrected.csv')
        
        self.processed_tensor, self.col_info, self.LOS = self.process()

    def __getitem__(self, index):
        """
        Returns a single sample and its label corresponding to the given index.

        Args:
            index (int): Index of the sample within the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of (input_tensor, y_label),
            where `input_tensor` is the input feature tensor and `y_label` is the
            corresponding label tensor.
        """
        input_tensor = self.processed_tensor[index, :-1]
        y_label = self.processed_tensor[index, -1]
        los = self.LOS[index]
        return input_tensor, y_label, los
    
    def __len__(self):
        """
        Returns:
            int: Size of the dataset (number of samples).
        """
        return self.processed_tensor.shape[0]
    
    def process(self):
        """
        Reads the raw data and performs preprocessing.

        Preprocessing steps:
        1. Load the CSV file ('missing_corrected.csv')
        2. Separate and remove the 'LOS' (Length of Stay) column
        3. Clean and organize the label column ('REASONb') using `organize_labels`
        4. Extract column metadata using `get_col_info`
        5. Convert the Pandas DataFrame to a PyTorch tensor using `df_to_tensor`

        Raises:
            ValueError: If the raw data does not contain the 'LOS' or 'REASONb' column.

        Returns:
            tuple: A tuple (df_tensor, col_info, LOS) containing the preprocessed data.
        """
        # missing value 
        df = tackle_missing_value_wrapper(self.raw_data_path, self.missing_corrected_path)

        # remove unused variables
        # 1. CASEID -> ID of the cases.
        # 2. DISYR -> same in all cases.
        # These two things aren't needed in training model.
        df = df.drop(['DISYR', 'CASEID'], axis=1)

        # get los
        if 'LOS' in df.columns:
            LOS = df['LOS']
            LOS = df_to_tensor(LOS)
            df = df.drop('LOS', axis=1)
        else:
            raise ValueError('No LOS variable in the raw data.')
        
        # Prepare REASON or REASONb
        if 'REASON' not in df.columns:
            raise ValueError('No REASON variable in the raw data.')
        
        if self.binary:
            df = make_binary(df)
        else:
            # rearrange the position of REASON column to easily get the y item like this:
            # ```y_label = self.processed_tensor[index, -1]```
            columns = list(df.columns)
            columns.remove('REASON')
            columns.append('REASON')
            df = df[columns]
        
        self.processed_df = df

        # To use torch.Embedding, organizing label as successive integers is needed.
        df = organize_labels(df, self.ig_label)
        # make pd.DataFrame into torch.Tensor.
        df_tensor = df_to_tensor(df)
        # get col infos, list of (col_list, col_dims, ad_col_index, dis_col_index)
        # ad_col_index, dis_col_index: integer position of admission col, discharge col
        if self.binary:
            self.num_classes = len(df["REASONb"].unique())
            df = df.drop("REASONb", axis=1)
            col_info = get_col_info(df, ig_label=self.ig_label)
        else:
            self.num_classes = len(df["REASON"].unique())
            df = df.drop("REASON", axis=1)
            col_info = get_col_info(df, ig_label=self.ig_label)

        # col_info: (col_list, col_dims, ad_col_index, dis_col_index)
        return df_tensor, col_info, LOS # -> self.process하면 tuple로 반환될 것
    
    
class TEDSDatasetForGIN(Dataset):
    def __init__(self, root, binary=True):
        """
        Constructor for the TEDSDatasetForGIN.
        This Dataset is for Plain GIN. Static graph representation. 
        (Does not seperate admission and discharge)

        This initializes dataset paths, creates required directories, and either loads
        previously processed data or performs a new preprocessing step and loads the
        resulting data into memory.

        Args:
            root (str): Root directory path where the dataset is stored.
            binary (bool): Whether to convert REASON into its binary form (REASONb).
        """
        super().__init__()
        self.binary = binary

        self.root = root
        self.raw_data_path = os.path.join(self.root, 'raw', 'TEDS_Discharge.csv')
        self.missing_corrected_path = os.path.join(self.root, 'raw', 'missing_corrected.csv')

        df = tackle_missing_value_wrapper(self.raw_data_path, self.missing_corrected_path)

        # remove unused variables
        # 1. CASEID -> ID of the cases.
        # 2. DISYR -> same in all cases.
        # These two things aren't needed in training model.
        df = df.drop(['DISYR', 'CASEID'], axis=1)

        if 'REASON' not in df.columns:
            raise ValueError('no \"REASON\" variable in the raw data.')
        
        if self.binary:
            df = make_binary(df)
        else:
            columns = list(df.columns)
            columns.remove('REASON')
            columns.append('REASON')
            df = df[columns]

        # label_organize
        df = organize_labels(df)
        self.processed_df = df
        
        # make pd.DataFrame into torch.Tensor.
        self.df_tensor = df_to_tensor(df)

        if self.binary:
            self.num_classes = len(df["REASONb"].unique())
            df = df.drop("REASONb", axis=1)
            self.col_dims = get_col_dims(df)
        else:
            self.num_classes = len(df["REASON"].unique())
            df = df.drop("REASON", axis=1)
            self.col_dims = get_col_dims(df)
        
        

    def __getitem__(self, index):
        """
        Returns a single sample and its label corresponding to the given index.

        Args:
            index (int): Index of the sample within the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of (input_tensor, y_label),
            where `input_tensor` is the input feature tensor and `y_label` is the
            corresponding label tensor.
        """
        x = self.df_tensor[index, :-1]
        y = self.df_tensor[index, -1]
        return x, y
    
    def __len__(self):
        """
        Returns:
            int: Size of the dataset (number of samples).
        """
        return self.df_tensor.shape[0]
    