import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sys

class TProv:
    def __init__(self,func):
        self.func = func
    def __call__(self,*args,**kwargs):
        method_name = self.func.__name__
        if method_name in ["query","filter"]:
            return self.decorate_filter(*args,**kwargs)
        elif method_name == "merge":
            return self.decorate_merge(*args,**kwargs)
        elif method_name=="concat":
            return self.decorate_concat(*args,**kwargs)
        elif method_name == "join":
            return self.decorate_join(*args,**kwargs)
        elif method_name == "drop":
            return self.decorate_vertical_reduction(*args,**kwargs)
        elif method_name == "get_dummies":
            return self.decorate_vertical_augmentation(*args,**kwargs)
        elif method_name in ["sample","dropna"]:
            return self.decorate_horizontal_reduction(*args,**kwargs)
        elif method_name in ["rename","apply","assign","replace","fillna"]:
            return self.decorate_transform(*args,**kwargs)
        else:
            raise NotImplementedError(f"No provenance tracking for '{method_name}'.")
    def decorate_filter(self,df,condition):
        out_df = df.query(condition)
        m = out_df.shape[0]
        n = df.shape[0]
        row_indices = range(m)
        col_indices = out_df.index
        data = np.ones(m)
        sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(m,n))
        return out_df, sparse_matrix
    def decorate_merge(self,df1,df2,**kwargs):
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        df1_copy["id1"] = range(len(df1))
        df2_copy["id2"] = range(len(df2))
        df_merge = pd.merge(df1_copy,df2_copy,**kwargs)
        left_provenance_matrix = np.zeros((len(df_merge),len(df1)))
        right_provenance_matrix = np.zeros((len(df_merge),len(df2)))
        for i,idx in enumerate(df_merge["id1"]):
            if pd.notna(idx):
                left_provenance_matrix[i,int(idx)] = 1
        for i,idx in enumerate(df_merge["id2"]):
            if pd.notna(idx):    
                right_provenance_matrix[i,int(idx)] = 1
        df_merge = df_merge.drop(columns=["id1","id2"])
        left_provenance_matrix = csr_matrix(left_provenance_matrix)
        right_provenance_matrix = csr_matrix(right_provenance_matrix)
        return df_merge, (left_provenance_matrix,right_provenance_matrix)
    def decorate_concat(self,dfs:list[pd.DataFrame],**kwargs):
        df_concat = pd.concat(dfs,**kwargs)
        len_max = max([len(df) for df in dfs])
        matrices = []
        axis = kwargs.get("axis", 0)
        if axis == 1:
            for df in dfs:
                m = len_max
                n = len(df)
                diag_values = np.ones(n)
                matrix = csr_matrix((diag_values, (range(n), range(n))), shape=(m, n))
                matrices.append(matrix)
        else :
            m = sum([len(df) for df in dfs])
            i = 0
            for df in dfs:
                n = len(df)
                diag_values = np.ones(n)
                matrix = csr_matrix((diag_values, (range(i,i+n), range(n))), shape=(m, n))
                matrices.append(matrix)
                i+= n
        return df_concat, matrices
    def decorate_join(self,df1,df2,**kwargs):
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        df1_copy["id1"] = range(len(df1))
        df2_copy["id2"] = range(len(df2))
        df_join = df1_copy.join(df2_copy, **kwargs)
        matrix1 = np.zeros((len(df_join),len(df1)))
        matrix2 = np.zeros((len(df_join),len(df2)))
        for i,idx in enumerate(df_join["id1"]):
            if pd.notna(idx):
                idx = int(idx)
                matrix1[i,idx] = 1
        for i,idx in enumerate(df_join["id2"]):
            if pd.notna(idx):  
                idx = int(idx)  
                matrix2[i,idx] = 1
        df_join = df_join.drop(columns=["id1","id2"])
        matrix1 = csr_matrix(matrix1)
        matrix2 = csr_matrix(matrix2)
        return df_join, (matrix1,matrix2)
    
    def decorate_vertical_reduction(self, df, columns=None, axis=1, **kwargs):
        """Vertical Reduction - suppression de colonnes (Feature Selection/Drop Columns)"""
        if columns is None:
            out_df = df.drop(axis=axis, **kwargs)
        else:
            out_df = df.drop(columns=columns, **kwargs)
        
        # Tenseur identité car même nombre de lignes
        n = len(df)
        identity_tensor = csr_matrix((np.ones(n), (range(n), range(n))), shape=(n, n))
        return out_df, identity_tensor
    
    def decorate_vertical_augmentation(self, df, columns=None, **kwargs):
        """Vertical Augmentation - ajout de colonnes (One-Hot Encoding)"""
        if hasattr(pd, 'get_dummies') and self.func.__name__ == 'get_dummies':
            # Pour pd.get_dummies
            if columns is None:
                out_df = pd.get_dummies(df, **kwargs)
            else:
                out_df = pd.get_dummies(df, columns=columns, **kwargs)
        else:
            # Pour d'autres opérations d'augmentation
            out_df = df.copy()
            # Ici on pourrait ajouter des colonnes selon les kwargs
        
        # Tenseur identité car même nombre de lignes
        n = len(df)
        identity_tensor = csr_matrix((np.ones(n), (range(n), range(n))), shape=(n, n))
        return out_df, identity_tensor
    
    def decorate_horizontal_reduction(self, df, n=None, frac=None, **kwargs):
        """Horizontal Reduction - réduction du nombre de lignes (sampling, filtering)"""
        # Correction :
        if self.func.__name__ == 'sample':
            out_df = df.sample(n=n, frac=frac, **kwargs)
        else:
            out_df = df.dropna(**kwargs)
        
        # Créer tenseur de mapping pour les indices survivants
        m = len(out_df)
        n_orig = len(df)
        
        if hasattr(out_df, 'index'):
            surviving_indices = out_df.index.tolist()
        else:
            surviving_indices = list(range(m))
        
        row_indices = range(m)
        col_indices = surviving_indices
        data = np.ones(m)
        
        mapping_tensor = csr_matrix((data, (row_indices, col_indices)), shape=(m, n_orig))
        return out_df, mapping_tensor
    def decorate_transform(self, df, *args, **kwargs):
        """Data Transformation - modification des valeurs sans changer les lignes"""
        # Appliquer la transformation
        out_df = self.func(df, *args, **kwargs)
        
        # Tenseur identité (même nombre de lignes)
        n = len(df)
        identity_tensor = csr_matrix((np.ones(n), (range(n), range(n))), shape=(n, n))
        return out_df, identity_tensor
def to_datetime_with_provenance(df, column):
    """Wrapper pour pd.to_datetime avec provenance"""
    df_copy = df.copy()
    df_copy[column] = pd.to_datetime(df_copy[column])
    n = len(df)
    identity_tensor = csr_matrix((np.ones(n), (range(n), range(n))), shape=(n, n))
    return df_copy, identity_tensor
def select_columns_with_provenance(df, columns):
    """Fonction wrapper pour sélection de colonnes avec provenance"""
    if isinstance(columns, str):
        out_df = df[[columns]]  # Forcer DataFrame même pour une colonne
    else:
        out_df = df[columns]
    
    # Tenseur identité (même nombre de lignes)
    n = len(df)
    identity_tensor = csr_matrix((np.ones(n), (range(n), range(n))), shape=(n, n))
    return out_df, identity_tensor
def assign_column_with_provenance(df, column_name, values):
    df_copy = df.copy()
    df_copy[column_name] = values
    n = len(df)
    identity_tensor = csr_matrix((np.ones(n), (range(n), range(n))), shape=(n, n))
    return df_copy, identity_tensor

def compose_all_tensors(*tensors):
    result = tensors[0]
    for tensor in tensors[1:]:
        result = tensor @ result
    return result

# Vos fonctions exactement comme elles sont
def get_sources_for_row(output_row_idx, tensor):
    if output_row_idx >= tensor.shape[0]:
        return []
    row = tensor.getrow(output_row_idx)
    _, input_indices = row.nonzero()
    return input_indices.tolist()

def get_destinations_for_row(input_row_idx, tensor):
    if input_row_idx >= tensor.shape[1]:
        return []
    col = tensor.getcol(input_row_idx)
    output_indices, _ = col.nonzero()
    return output_indices.tolist()

def query_merge_provenance(output_row_idx, tensor_tuple):
    tensor1, tensor2 = tensor_tuple
    left_sources = get_sources_for_row(output_row_idx, tensor1)
    right_sources = get_sources_for_row(output_row_idx, tensor2)
    return {
        'left_dataframe_sources': left_sources,
        'right_dataframe_sources': right_sources
    }

def trace_provenance_through_pipeline(final_row_idx, *tensors):
    lineage = []
    current_row_idx = final_row_idx
    for i, tensor in enumerate(reversed(tensors)):
        if isinstance(tensor, tuple):
            # Gestion des opérations merge/join 
            sources = query_merge_provenance(current_row_idx, tensor)
            lineage.append(sources)
            # Prendre la première source disponible pour continuer
            all_sources = sources['left_dataframe_sources'] + sources['right_dataframe_sources']
            if all_sources:
                current_row_idx = all_sources[0]
        else:
            sources = get_sources_for_row(current_row_idx, tensor)
            lineage.append(sources)
            if sources and len(sources) > 0:
                current_row_idx = sources[0]  # Prendre le premier élément de la liste
            else:
                break  # Arrêter si pas de sources
    return lineage[::-1]