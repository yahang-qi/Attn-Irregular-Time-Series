import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


static_vars = ['Age', 'Gender', 'Height', 'Weight', 'ICUType']
temporal_vars = ['Temp', 'pH', 'FiO2', 'TroponinT', 'Creatinine', 'PaCO2', 'HCT', 'TroponinI',\
    'AST', 'Mg', 'SysABP', 'RespRate', 'NIDiasABP', 'Platelets', 'Cholesterol', 'Albumin', 'MechVent',\
    'NISysABP', 'Glucose', 'MAP', 'ALT', 'Lactate', 'Na', 'K', 'WBC', 'SaO2', 'HCO3', 'Bilirubin',\
    'BUN', 'ALP', 'Weight', 'DiasABP', 'PaO2', 'Urine', 'HR', 'GCS', 'NIMAP']


class PysioNet2012Dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')]
        
        # load max length of each variable for padding
        path_max_lens = os.path.join(path, 'max_lens.csv')
        max_lens = {}
        
        with open(path_max_lens, 'r') as fp:
            for line in fp:
                var, max_len = line.strip().split(',')
                max_lens[var] = int(max_len)
            fp.close()
            
        self.max_lens = max_lens
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with open(self.files[idx], 'r') as fp:
            record = json.load(fp)
            fp.close()
        
        static_features = [record['Static'][var] for var in static_vars]
        temporal_features = {var: [record['Temporal'][var]['Time'], record['Temporal'][var]['Value']] for var in temporal_vars}
        
        dict_temporal = {}
        for var, values in temporal_features.items():
            values_tensor = torch.tensor(values, dtype=torch.float32).T
            # pad the tensor with -1.0 to the max length
            pad_len = self.max_lens[var] - values_tensor.size(0)
            if pad_len > 0:
                pad = -1.0 * torch.ones(pad_len, values_tensor.size(1))
                values_tensor = torch.cat([values_tensor, pad], dim=0)
            dict_temporal[var] = values_tensor
        
        item = {
            'Static': torch.tensor(static_features, dtype=torch.float32),
            'Temporal': dict_temporal,
            'Outcome': torch.tensor(record['Outcome'], dtype=torch.float32)
        }
            
        return item        
        
        
if __name__ == '__main__':
    dataset = PysioNet2012Dataset('./datasets/PysioNet2012_Preprocessed/set-a')
    
    print(dataset[0])
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
    )
    for i in loader:
        print(i)
        break
    