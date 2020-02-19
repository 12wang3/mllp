import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from mllp.discretizer import MinimalEntropyDiscretizer


def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    return f_list[:-1], int(f_list[-1][-1])


def read_csv(data_path, info_path, shuffle=False):
    D = pd.read_csv(data_path, header=None)
    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    f_list, label_pos = read_info(info_path)
    f_df = pd.DataFrame(f_list)
    D.columns = f_df.iloc[:, 0]
    y_df = D.iloc[:, [label_pos]]
    X_df = D.drop(D.columns[label_pos], axis=1)
    f_df = f_df.drop(f_df.index[label_pos])
    return X_df, y_df, f_df, label_pos


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False):
        self.f_df = f_df
        self.discrete = discrete
        self.label_enc = preprocessing.OneHotEncoder(categories='auto')
        self.me_discretizer = MinimalEntropyDiscretizer()
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto')
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = None
        self.y_fname = None

    def split_data(self, X_df):
        discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(np.float)
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        self.label_enc.fit(y_df)
        self.y_fname = list(self.label_enc.get_feature_names(y_df.columns))

        if not continuous_data.empty:
            if self.discrete:
                self.me_discretizer = MinimalEntropyDiscretizer()
                self.me_discretizer.fit(continuous_data, y_df)
                isna_df = continuous_data.isna()
                continuous_data = self.me_discretizer.transform(continuous_data)
                for k in isna_df:
                    continuous_data.loc[isna_df[k], k] = '?'
                discrete_data = pd.concat([discrete_data, continuous_data], axis=1)
            else:
                # Use mean as missing value for continuous columns if do not discretize them.
                self.imp.fit(continuous_data.values)
        if not discrete_data.empty:
            # One-hot encoding
            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            self.X_fname = list(self.feature_enc.get_feature_names(feature_names))
            if not self.discrete:
                self.X_fname.extend(continuous_data.columns)
        else:
            self.X_fname = continuous_data.columns

    def transform(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        # Encode string value to int index.
        y = self.label_enc.transform(y_df.values.reshape(-1, 1)).toarray()

        if not continuous_data.empty:
            if self.discrete:
                isna_df = continuous_data.isna()
                continuous_data = self.me_discretizer.transform(continuous_data)
                for k in isna_df:
                    continuous_data.loc[isna_df[k], k] = '?'
                discrete_data = pd.concat([discrete_data, continuous_data], axis=1)
            else:
                # Use mean as missing value for continuous columns if we do not discretize them.
                continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                               columns=continuous_data.columns)
        if not discrete_data.empty:
            # One-hot encoding
            discrete_data = self.feature_enc.transform(discrete_data)
            if not self.discrete:
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            else:
                X_df = pd.DataFrame(discrete_data.toarray())
        else:
            X_df = continuous_data
        return X_df.values, y


class UnionFind:
    """Union-Find algorithm used for merging the identical nodes in MLLP."""

    def __init__(self, keys):
        self.stu = {}
        for k in keys:
            self.stu[k] = k

    def find(self, x):
        try:
            self.stu[x]
        except KeyError:
            return x
        if x != self.stu[x]:
            self.stu[x] = self.find(self.stu[x])
        return self.stu[x]

    def union(self, x, y):
        xf = self.find(x)
        yf = self.find(y)
        if xf != yf:
            self.stu[yf] = xf
            return True
        return False
