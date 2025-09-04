import pandas as pd
import numpy as np
import time
import argparse
import os
from TProv import TProv, select_columns_with_provenance,assign_column_with_provenance,compose_all_tensors
from scipy.sparse import csr_matrix

def assign_column_with_provenance(df, column_name, values):
    df_copy = df.copy()
    df_copy[column_name] = values
    n = len(df)
    identity_tensor = csr_matrix((np.ones(n), (range(n), range(n))), shape=(n, n))
    return df_copy, identity_tensor

def main():
    input_path = './Datasets/census.csv'
    filename_ext = os.path.basename(input_path)
    filename, ext = os.path.splitext(filename_ext)
    output_path = 'processed_results'
    savepath = os.path.join(output_path, filename)
    os.makedirs(savepath, exist_ok=True)
    
    df = pd.read_csv(input_path)
    tensors = []
    
    # Assign names to columns
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'label']
    df.columns = names
    
    print("Initial dataset preview:")
    print(df.head())
    print('[' + time.strftime("%d/%m-%H:%M:%S") + '] Processing started')
    
    # OPERATION 0: Cleanup names from spaces
    col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'label']
    for c in col:
        cleaned_values = df[c].map(str.strip)
        df, tensor = assign_column_with_provenance(df, c, cleaned_values)
        tensors.append(tensor)
    
    # OPERATION 1: Replace ? character with NaN
    replace_prov = TProv(pd.DataFrame.replace)
    df, tensor1 = replace_prov(df, '?', np.nan)
    tensors.append(tensor1)
    
    # OPERATION 2-3: One-hot encode categorical variables
    col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    dummies_prov = TProv(pd.get_dummies)
    df, tensor2 = dummies_prov(df, columns=col)
    tensors.append(tensor2)
    
    # OPERATION 4: Assign sex and label binary values
    sex_values = df['sex'].replace({'Male': 1, 'Female': 0})
    df, tensor4a = assign_column_with_provenance(df, 'sex', sex_values)
    tensors.append(tensor4a)
    
    label_values = df['label'].replace({'<=50K': 0, '>50K': 1})
    df, tensor4b = assign_column_with_provenance(df, 'label', label_values)
    tensors.append(tensor4b)
    
    # OPERATION 5: Drop fnlwgt variable
    drop_prov = TProv(pd.DataFrame.drop)
    df, tensor5 = drop_prov(df, columns=['fnlwgt'])
    tensors.append(tensor5)
    
    print("Processed dataset preview:")
    print(df.head())
    
    # Save processed dataset
    df.to_csv(os.path.join(savepath, 'census_processed.csv'), index=False)
    print('[' + time.strftime("%d/%m-%H:%M-%S") + '] Processing completed')
    
    
    result_tensor = compose_all_tensors(*tensors)
    return tensors, result_tensor

if __name__ == '__main__':
    final_tensors, result_tensor = main()
    print(f"Nombre d'éléments non-zéros: {result_tensor.nnz}")
    print(f"Dimensions du tenseur résultat: {result_tensor.shape}")