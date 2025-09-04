import sys
import pandas as pd
import numpy as np
import argparse
import os
from TProv import TProv,assign_column_with_provenance,compose_all_tensors
from scipy.sparse import csr_matrix
import torch
from scipy.sparse import eye


def main(opt):
    input_path = './Datasets/german.csv'

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at {input_path}")

    df = pd.read_csv(input_path, header=0)
    tensors = []

    print("\nInitial dataset preview:")
    print(df.head())

    # OPERATION 0: Replace cryptic values with meaningful labels
    replace_prov = TProv(pd.DataFrame.replace)
    df, tensor0 = replace_prov(df, {'checking': {'A11': 'check_low', 'A12': 'check_mid', 'A13': 'check_high', 'A14': 'check_none'},
                     'credit_history': {'A30': 'debt_none', 'A31': 'debt_noneBank', 'A32': 'debt_onSchedule', 'A33': 'debt_delay', 'A34': 'debt_critical'},
                     'purpose': {'A40': 'pur_newCar', 'A41': 'pur_usedCar', 'A42': 'pur_furniture', 'A43': 'pur_tv',
                                 'A44': 'pur_appliance', 'A45': 'pur_repairs', 'A46': 'pur_education', 'A47': 'pur_vacation',
                                 'A48': 'pur_retraining', 'A49': 'pur_business', 'A410': 'pur_other'},
                     'savings': {'A61': 'sav_small', 'A62': 'sav_medium', 'A63': 'sav_large', 'A64': 'sav_xlarge', 'A65': 'sav_none'},
                     'employment': {'A71': 'emp_unemployed', 'A72': 'emp_lessOne', 'A73': 'emp_lessFour', 'A74': 'emp_lessSeven', 'A75': 'emp_moreSeven'},
                     'other_debtors': {'A101': 'debtor_none', 'A102': 'debtor_coApp', 'A103': 'debtor_guarantor'},
                     'property': {'A121': 'prop_realEstate', 'A122': 'prop_agreement', 'A123': 'prop_car', 'A124': 'prop_none'},
                     'other_inst': {'A141': 'oi_bank', 'A142': 'oi_stores', 'A143': 'oi_none'},
                     'housing': {'A151': 'hous_rent', 'A152': 'hous_own', 'A153': 'hous_free'},
                     'job': {'A171': 'job_unskilledNR', 'A172': 'job_unskilledR', 'A173': 'job_skilled', 'A174': 'job_highSkill'},
                     'phone': {'A191': 0, 'A192': 1},
                     'foreigner': {'A201': 1, 'A202': 0},
                     'label': {2: 0}})
    tensors.append(tensor0)

    # OPERATION 1: Map gender and marital status
    status_values = df['personal_status'].map({'A91': 'divorced', 'A92': 'divorced', 'A93': 'single', 'A95': 'single'}).fillna('married')
    df, tensor1 = assign_column_with_provenance(df, 'status', status_values)
    tensors.append(tensor1)
    
    gender_values = df['personal_status'].map({'A92': 0, 'A95': 0}).fillna(1)
    df, tensor2 = assign_column_with_provenance(df, 'gender', gender_values)
    tensors.append(tensor2)

    # OPERATION 2: Drop the original 'personal_status' column
    drop_prov = TProv(pd.DataFrame.drop)
    df, tensor3 = drop_prov(df, columns=['personal_status'])
    tensors.append(tensor3)

    # OPERATION 3-13: One-hot encode categorical columns
    categorical_cols = ['checking', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property',
                        'other_inst', 'housing', 'job', 'status']
    
    dummies_prov = TProv(pd.get_dummies)
    df, tensor4 = dummies_prov(df, columns=categorical_cols)
    tensors.append(tensor4)

    print("\nProcessed dataset preview:")
    print(df.head())
    print("\nPipeline execution completed successfully.")
    
    return tensors,compose_all_tensors(*tensors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-op', dest='opt', action='store_true', help='Use the optimized capture')
    args = parser.parse_args()
    final_tensors,final_tensor = main(args.opt)
    print(f"Nombre d'éléments non-zéros: {final_tensor.nnz}")
    print(f"Dimensions du tenseur résultat: {final_tensor.shape}")

