import pandas as pd
import numpy as np
import pprint as pp
from scipy.sparse import csr_matrix

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
            #return self.decorate_concat(*args,**kwargs)
        elif method_name == "join":
            return self.decorate_join(*args,**kwargs)
        else:
            raise NotImplementedError(f"No provenance tracking for '{method_name}'.")
    def decorate_filter(self,df,condition):
        out_df = df.query(condition)
        m = out_df.shape[0]
        n = df.shape[0]
        matrix = np.zeros((m,n))
        for i,idx in enumerate(out_df.index):
            print(idx)
            matrix[i,idx] = 1
        matrix = csr_matrix(matrix)
        return out_df,matrix
    def decorate_merge(self,df1,df2,**kwargs):
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        df1_copy["id1"] = range(len(df1))
        df2_copy["id2"] = range(len(df2))
        df_merge = pd.merge(df1_copy,df2_copy,**kwargs)
        matrix1 = np.zeros((len(df_merge),len(df1)))
        matrix2 = np.zeros((len(df_merge),len(df2)))
        for i,idx in enumerate(df_merge["id1"]):
            if pd.notna(idx):
                matrix1[i,idx] = 1
        for i,idx in enumerate(df_merge["id2"]):
            if pd.notna(idx):    
                matrix2[i,idx] = 1
        df_merge = df_merge.drop(columns=["id1","id2"])
        matrix1 = csr_matrix(matrix1)
        matrix2 = csr_matrix(matrix2)
        return df_merge, (matrix1,matrix2)
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
                #return df_concat, (csr_matrix(np.eye()))
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
                idx = int(idx)  # Ensure idx is an integer
                matrix1[i,idx] = 1
        for i,idx in enumerate(df_join["id2"]):
            if pd.notna(idx):  
                idx = int(idx)  # Ensure idx is an integer  
                matrix2[i,idx] = 1
        df_join = df_join.drop(columns=["id1","id2"])
        matrix1 = csr_matrix(matrix1)
        matrix2 = csr_matrix(matrix2)
        return df_join, (matrix1,matrix2)
    # def decorate_merge(self, df1, df2, **kwargs):
    #     df1_copy = df1.copy()
    #     df2_copy = df2.copy()
    #     df1_copy["_id1"] = range(len(df1))
    #     df2_copy["_id2"] = range(len(df2))
    #     df_out = pd.merge(df1_copy, df2_copy, **kwargs)

    #     idx1 = df_out["_id1"].values
    #     idx2 = df_out["_id2"].values

    #     n = len(df_out)
    #     tensor1 = csr_matrix((np.ones(n), (range(n), idx1)), shape=(n, len(df1)))
    #     tensor2 = csr_matrix((np.ones(n), (range(n), idx2)), shape=(n, len(df2)))
    #     return df_out.drop(columns=["_id1", "_id2"]), (tensor1, tensor2)
# df = pd.DataFrame({"A":[1,2,4],"B":[9,5,np.nan]})
# condition = "A>1"
# filter_with_prov = TProv(pd.DataFrame.query)
# df_filt,matrix=filter_with_prov(df,condition)

# print(df_filt,matrix,df)

# print((i,idx) for i,idx in enumerate(df))

# print(df.loc[0:2,"A"])
# print(df[[True,False,True]].index)
# for i,idx in enumerate(df):
#     print(i,idx)
# print((i,idx) for i,idx in enumerate(df.index))
# print(df.query("A>2"))
df1 = pd.DataFrame({"ID":[10,20,30,40],"Birthdate":[1996,1994,np.nan,1982],"Gender":['F','M','F','M'],"Postcode":[1234,1234,1234,1234]})
df2 = pd.DataFrame({"Name":["Alice","Bob"]})
# merge_with_prov = TProv(pd.merge)
# df_merge,tensor = merge_with_prov(df1,df2,on="ID",how="inner")
# tensor1,tensor2 = tensor
# print(tensor1)
# print(tensor2)
# print(csr_matrix((np.ones(5),([0,1,2,3,4],[1,4,5,6,8]))))
join_with_prov = TProv(pd.DataFrame.join)
df_concat, tensors = join_with_prov(df1, df2)
tensor1, tensor2 = tensors
print(tensor1)
print(tensor2)
print(df_concat)