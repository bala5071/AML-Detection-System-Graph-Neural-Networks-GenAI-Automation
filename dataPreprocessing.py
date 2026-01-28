import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import LabelEncoder

class Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['HI-Small_Trans.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']


    def get_all_account(self, df):
        ldf = df[['Account', 'From Bank']]
        rdf = df[['Account.1', 'To Bank']]

        suspicious = df[df['Is Laundering'] == 1]
        s1 = suspicious[['Account', 'Is Laundering']]
        s2 = suspicious[['Account.1', 'Is Laundering']]
        s2 = s2.rename({'Account.1': 'Account'}, axis=1)

        suspicious = pd.concat([s1, s2], join='outer').drop_duplicates()

        ldf = ldf.rename({'From Bank': 'Bank'}, axis=1)
        rdf = rdf.rename({'Account.1': 'Account', 'To Bank': 'Bank'}, axis=1)

        all_accounts = pd.concat([ldf, rdf], join='outer').drop_duplicates()
        all_accounts['Is Laundering'] = 0
        all_accounts.set_index('Account', inplace=True)
        all_accounts.update(suspicious.set_index('Account'))
        all_accounts = all_accounts.reset_index()
        
        return all_accounts

    def calculate_unique_counterparties(self, df, accounts_df):
        out_degree = df.groupby('Account')['Account.1'].nunique()
 
        in_degree = df.groupby('Account.1')['Account'].nunique()

        degrees = out_degree.add(in_degree, fill_value=0)

        accounts_df['unique_counterparties'] = accounts_df['Account'].map(degrees).fillna(0)
        return accounts_df

    def process(self):
        print("Processing raw data...")
        df = pd.read_csv(self.raw_paths[0])

        le_cols = ['Payment Format', 'Payment Currency', 'Receiving Currency']
        le = LabelEncoder()
        for col in le_cols:
            df[col] = le.fit_transform(df[col].astype(str))

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        

        df = df.sort_values(['Account', 'Account.1', 'Timestamp'])

        df['Time Delta'] = df.groupby(['Account', 'Account.1'])['Timestamp'].diff().dt.total_seconds().fillna(0)

        time_numer = df['Timestamp'].apply(lambda x: x.value)
        df['Timestamp_Norm'] = (time_numer - time_numer.min()) / (time_numer.max() - time_numer.min())


        df['Account'] = df['From Bank'].astype(str) + '_' + df['Account']
        df['Account.1'] = df['To Bank'].astype(str) + '_' + df['Account.1']


        node_df = self.get_all_account(df)
        
        node_df = self.calculate_unique_counterparties(df, node_df)
        


        out_txn_count = df.groupby('Account').size()
        in_txn_count = df.groupby('Account.1').size()

        node_df['out_txn_count'] = node_df['Account'].map(out_txn_count).fillna(0)
        node_df['in_txn_count'] = node_df['Account'].map(in_txn_count).fillna(0)

        out_amount_sum = df.groupby('Account')['Amount Paid'].sum()
        in_amount_sum = df.groupby('Account.1')['Amount Received'].sum()

        node_df['out_amount_sum'] = node_df['Account'].map(out_amount_sum).fillna(0)
        node_df['in_amount_sum'] = node_df['Account'].map(in_amount_sum).fillna(0)

        node_df['flow_imbalance'] = (
            node_df['out_amount_sum'] - node_df['in_amount_sum']
        )

        for col in [
            'out_txn_count', 'in_txn_count',
            'out_amount_sum', 'in_amount_sum',
            'flow_imbalance'
        ]:
            node_df[col] = np.log1p(node_df[col].abs())
        

        currency_ls = sorted(df['Receiving Currency'].unique())
        paying_df = df[['Account', 'Amount Paid', 'Payment Currency']]
        receiving_df = df[['Account.1', 'Amount Received', 'Receiving Currency']].rename({'Account.1': 'Account'}, axis=1)
        
        node_df = node_df.fillna(0)
        node_df = node_df.drop(columns=['Bank'])


        y = torch.from_numpy(node_df['Is Laundering'].values).long()

        x_df = node_df.drop(['Account', 'Is Laundering'], axis=1)
        x = torch.from_numpy(x_df.values).float()

        node_mapping = {name: i for i, name in enumerate(node_df['Account'])}
        
        df['Src'] = df['Account'].map(node_mapping)
        df['Dst'] = df['Account.1'].map(node_mapping)

        edge_index = torch.stack([
            torch.from_numpy(df['Src'].values), 
            torch.from_numpy(df['Dst'].values)
        ], dim=0).long()
        df['Log Amount'] = np.log1p(df['Amount Paid'])

        edge_attr_cols = [
            'Timestamp_Norm', 
            'Amount Received', 
            'Receiving Currency', 
            'Amount Paid', 
            'Payment Currency', 
            'Payment Format',
            'Log Amount',       
            'Time Delta'        
        ]
        
        edge_attr = torch.from_numpy(df[edge_attr_cols].values).float()
        log_amount = edge_attr[:, 6]
        mean = log_amount.mean()
        std = log_amount.std()

        edge_weight = (log_amount - mean) / (std + 1e-8)
        edge_weight = edge_weight.clamp(-10, 10)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr, y=y)
        if self.pre_filter is not None and not self.pre_filter(data):
            return
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        print("Data processing complete. Saved to data.pt")


