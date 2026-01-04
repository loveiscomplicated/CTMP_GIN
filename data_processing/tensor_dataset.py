'''
결국 pyg Data / Batch를 생성해서 저장 또는 로드할 필요가 없었고, 오히려 불편함, 저장공간 낭비였음
37개의 타임스탬프를 미리 만들어 저장하는 것은 원본 데이터의 37배에 달하는 용량으로 뻥튀기가 되는 것
우리는 어차피 필요한 게 1. 입소 시, 2. 퇴소 시, 3. 제로 패딩 이기 때문에
1,2만 들고 엔티티 임베딩까지 한 뒤에
모델 입력하기 직전에 시계열 데이터로 변환하면 됨

이 모든 걸 텐서로만 진행할 수 있음 + pyg Data / Batch를 쓰면 기본 알고리즘 때문에 쉐입이 이상해지던가 객체를 생성해서 저장하기 때문에 용량이 더 커짐


독스트링은 LLM에 의해 생성됨
'''
import os
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from data_utils import get_col_info, organize_labels, df_to_tensor, get_col_dims

CURDIR = os.path.dirname(__file__)

class TEDSTensorDataset(Dataset):
    """
    TEDS (Temporal Embedding Deep Sequence) 모델을 위한 PyTorch Dataset.

    원본 데이터를 pyg.Data/Batch 대신 **순수 PyTorch Tensor** 형태로 변환하고 저장하여,
    저장 공간 낭비와 불필요한 객체 생성 오버헤드를 방지합니다. 데이터에는 입소(Admission) 
    시점과 퇴소(Discharge) 시점 데이터만이 포함됩니다.

    Attributes:
        root (str): 데이터 저장 및 로드 경로의 루트 디렉토리.
        processed_tensor (torch.Tensor): 전처리된 최종 데이터 텐서 (입력 데이터 + 레이블). 
                                         Shape: (num_samples, num_features).
                                         DataLoader에게 이걸 X, y로 나누어 전달하게 됨
        col_info (tuple[list[int], list[int]]): 컬럼 정보. 
                                                (입소 시점 컬럼 인덱스 리스트, 퇴소 시점 컬럼 인덱스 리스트)
        LOS (pandas.Series): Length of Stay (재원 일수) 정보.
    """
    def __init__(self, root):
        """
        TEDSTensorDataset의 생성자.

        데이터 경로 설정, 디렉토리 생성 후, 전처리된 데이터를 로드하거나
        새로운 전처리 과정을 수행하여 데이터를 메모리에 로드합니다.

        Args:
            root: 데이터가 위치할 루트 디렉토리 경로.
        """
        super().__init__()
        self.root = root
        self.raw_dir = os.path.join(root, "raw")
        if not os.path.exists(self.raw_dir):
            os.mkdir(self.raw_dir)

        self.process_dir = os.path.join(root, 'process')
        if not os.path.exists(self.process_dir):
            os.mkdir(self.process_dir)

        processed_data_path = os.path.join(self.process_dir, "processed_data.pt")
        if os.path.exists(processed_data_path):
            print("저장되어 있는 전처리된 데이터가 있습니다. 해당 데이터를 불러오는 중..")
            self.processed_tensor, self.col_info, self.LOS = torch.load(processed_data_path, weights_only=False)
            print("불러오기 완료")
        else:
            print("저장되어 있는 전처리된 데이터가 없으므로, 전처리 과정을 진행합니다. 전처리된 데이터는 저장됩니다.")
            print(f"저장 경로: {processed_data_path}")
            processed_data = self.process
            print("전처리 완료")
            self.processed_tensor, self.col_info, self.LOS = processed_data
            print("불러오기 완료")
            torch.save(processed_data, processed_data_path)
            print("전처리된 데이터 저장 완료")

    def __getitem__(self, index):
        """
        주어진 인덱스에 해당하는 하나의 샘플과 레이블을 반환합니다.

        Args:
            index: 데이터 셋 내의 샘플 인덱스.

        Returns:
            (input_tensor, y_label): 입력 텐서와 레이블 텐서의 튜플.
        """
        input_tensor = self.processed_tensor[index, :-1]
        y_label = self.processed_tensor[index, -1]
        los = self.LOS[index]
        return input_tensor, y_label, los
    
    def __len__(self):
        """
        데이터 셋의 전체 샘플 개수를 반환합니다.

        Returns:
            데이터 셋의 크기 (샘플 개수).
        """
        return self.processed_tensor.shape[0]
    
    @property
    def process(self):
        """
        원본 데이터를 읽고 전처리 과정을 수행합니다.

        전처리 단계:
        1. CSV 파일 로드 ('missing_corrected.csv')
        2. 'LOS' (Length of Stay) 컬럼 분리 및 제거
        3. 레이블 ('REASONb') 정리 (organize_labels)
        4. 컬럼 정보 추출 (get_col_info)
        5. Pandas DataFrame을 PyTorch Tensor로 변환 (df_to_tensor)

        Raises:
            ValueError: 원본 데이터에 'LOS' 또는 'REASONb' 컬럼이 없을 경우 발생.

        Returns:
            (df_tensor, col_info, LOS): 전처리된 데이터 튜플.
        """
        data_path = os.path.join(self.raw_dir, 'missing_corrected.csv')
        df = pd.read_csv(data_path)

        # los 따로 빼기
        if 'LOS' in df.columns:
            LOS = df['LOS']
            LOS = df_to_tensor(LOS)
            df = df.drop('LOS', axis=1)
        else:
            raise ValueError('raw data에서 LOS 데이터를 찾을 수 없습니다.')
        
        if 'REASONb' not in df.columns:
            raise ValueError('raw data에서 REASONb 데이터를 찾을 수 없습니다.')
        
        # label_organize
        df = organize_labels(df)
        
        # df to tensor
        df_tensor = df_to_tensor(df)
        
        # get col infos, list of (col_list, col_dims, ad_col_index, dis_col_index)
        # ad_col_index, dis_col_index는 다음과 같음 integer position of admission col, discharge col
        df = df.drop("REASONb", axis=1)
        col_info = get_col_info(df)

        # processed_data는 (tensor, col_info, LOS)형태 
        # LOS는 pd.Series임
        # col_info는 다음과 같음 (col_list, col_dims, ad_col_index, dis_col_index)
        return df_tensor, col_info, LOS # -> self.process하면 tuple로 반환될 것
    
class TEDSDatasetForGIN(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.raw_dir = os.path.join(root, "raw")
        if not os.path.exists(self.raw_dir):
            os.mkdir(self.raw_dir)

        self.process_dir = os.path.join(root, 'process')
        if not os.path.exists(self.process_dir):
            os.mkdir(self.process_dir)

        data_path = os.path.join(self.raw_dir, 'missing_corrected.csv')
        df = pd.read_csv(data_path)
        
        if 'REASONb' not in df.columns:
            raise ValueError('raw data에서 REASONb 데이터를 찾을 수 없습니다.')
        
        # label_organize
        df = organize_labels(df)

        self.df_tensor = df_to_tensor(df)
        df = df.drop("REASONb", axis=1)
        self.col_dims = get_col_dims(df)
    
    def __getitem__(self, index):
        x = self.df_tensor[index, :-1]
        y = self.df_tensor[index, -1]
        return x, y
    
    def __len__(self):
        return self.df_tensor.shape[0]
    

