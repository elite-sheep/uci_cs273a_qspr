import deepchem as dc
from rdkit import Chem
import pandas as pd
import numpy as np

url = 'https://github.com/elite-sheep/uci_cs273a_qspr/blob/master/ESI01.xlsx?raw=true'
#xls = pd.ExcelFile('ESI01.xlsx')
df_orig = pd.read_excel(url, 'S6 | Modeling - reference term')
df = df_orig.loc[df_orig['In model'] == True]
df = df.iloc[:,[0,1,2,6,10]]

df_smiles_o = pd.read_excel(url, 'S2 | Ions').loc[:,['Abbreviation','SMILES']]

df_merge1 = df.merge(df_smiles_o, left_on='Cation',right_on='Abbreviation',how='left')
df_merge1 = df_merge1.rename(columns={"SMILES":"Cation SMILES"})
df_merge1 = df_merge1.drop(columns=['Abbreviation'])
df_merge2 = df_merge1.merge(df_smiles_o, left_on='Anion',right_on='Abbreviation',how='left')
df_merge2 = df_merge2.rename(columns={"SMILES":"Anion SMILES"})
df_1 = df_merge2.drop(columns=['Abbreviation'])
df_1 = df_1.rename(columns={df.columns[2]: 'Viscosity'})

train_cation_smiles = df_1['Cation SMILES'].loc[df_1['Subset'] == 'train'].tolist()
train_anion_smiles = df_1['Anion SMILES'].loc[df_1['Subset'] == 'train'].tolist()
test_cation_smiles = df_1['Cation SMILES'].loc[df_1['Subset'] == 'test'].tolist()
test_anion_smiles = df_1['Anion SMILES'].loc[df_1['Subset'] == 'test'].tolist()

print(len(train_cation_smiles))
print(len(test_cation_smiles))

y_train = df_1['Viscosity'].loc[df_1['Subset'] == 'train'].to_numpy()
y_test = df_1['Viscosity'].loc[df_1['Subset'] == 'test'].to_numpy()

smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O','CC(=O)'] # test smiles
featurizer = dc.feat.SmilesToSeq(None, max_len=32)
features = featurizer.featurize(smiles)
print(features.shape)
print(featurizer.descriptors)
print(len(featurizer.descriptors))

X_cation_train = featurizer.featurize(train_cation_smiles)
X_anion_train = featurizer.featurize(train_anion_smiles)

X_train = np.hstack((X_cation_train,X_anion_train))

print(X_cation_train.shape)
print(X_train.shape)

X_cation_test = featurizer.featurize(test_cation_smiles)
X_anion_test = featurizer.featurize(test_anion_smiles)

X_test = np.hstack((X_cation_test,X_anion_test))

print(X_test.shape)
