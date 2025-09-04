import sys
import pandas as pd
import numpy as np
import time
import argparse
import os
from TProv import TProv, select_columns_with_provenance, to_datetime_with_provenance,assign_column_with_provenance,compose_all_tensors
from scipy.sparse import csr_matrix

def main(opt):
    input_path = './Datasets/compas.csv'

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at {input_path}")

    df = pd.read_csv(input_path, header=0)
    tensors = []

    print("\nInitial dataset preview:")
    print(df.head())

    # OPERATION 0: Select relevant columns
    df, tensor0 = select_columns_with_provenance(df, ['age', 'c_charge_degree', 'race', 'sex', 'priors_count', 
             'days_b_screening_arrest', 'two_year_recid', 'c_jail_in', 'c_jail_out'])
    tensors.append(tensor0)

    # OPERATION 1: Remove missing values
    dropna_prov = TProv(pd.DataFrame.dropna)
    df, tensor1 = dropna_prov(df)
    tensors.append(tensor1)

    # OPERATION 2: Make race binary (1 = Caucasian, 0 = Other)
    race_values = df['race'].apply(lambda r: 1 if r == 'Caucasian' else 0)
    df, tensor2 = assign_column_with_provenance(df, 'race', race_values)
    tensors.append(tensor2)

    # OPERATION 3: Make 'two_year_recid' the label and reverse values 
    rename_prov = TProv(pd.DataFrame.rename)
    df, tensor3 = rename_prov(df, columns={'two_year_recid': 'label'})
    tensors.append(tensor3)
    
    label_values = df['label'].apply(lambda l: 0 if l == 1 else 1)
    df, tensor3b = assign_column_with_provenance(df, 'label', label_values)
    tensors.append(tensor3b)

    # OPERATION 4: Convert jail time to days
    df, tensor4a = to_datetime_with_provenance(df, 'c_jail_out')
    df, tensor4b = to_datetime_with_provenance(df, 'c_jail_in')
    jailtime_values = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df, tensor4c = assign_column_with_provenance(df, 'jailtime', jailtime_values)
    tensors.extend([tensor4a, tensor4b, tensor4c])

    # OPERATION 5: Drop jail in and out dates
    drop_prov = TProv(pd.DataFrame.drop)
    df, tensor5 = drop_prov(df, columns=['c_jail_in', 'c_jail_out'])
    tensors.append(tensor5)

    # OPERATION 6: Convert charge degree to binary (1 = Felony, 0 = Misdemeanor)
    charge_values = df['c_charge_degree'].apply(lambda s: 1 if s == 'F' else 0)
    df, tensor6 = assign_column_with_provenance(df, 'c_charge_degree', charge_values)
    tensors.append(tensor6)

    print("\nProcessed dataset preview:")
    print(df.head())
    print("\nPipeline execution completed successfully.")
    
    return tensors, compose_all_tensors(*tensors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-op', dest='opt', action='store_true', help='Use the optimized capture')
    args = parser.parse_args()
    final_tensors,result_tensor = main(args.opt)
    print(len(result_tensor.nonzero()[0]))
    