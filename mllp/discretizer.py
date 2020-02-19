import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict


class MinimalEntropyDiscretizer:
    """Discretizer using recursive minimal entropy partitioning algorithm.

    Partition one feature recursively by searching the partition boundary which minimizes the class information entropy
    of candidate partitions. Minimal Description Length Principle is used to determine the stopping criteria."""

    def __init__(self):
        self.boundaries = None

    def fit(self, continuous_data, y):
        self.boundaries = defaultdict(list)
        for k in continuous_data:
            now_df = pd.concat([continuous_data[k], y], axis=1).sort_values(by=[k]).dropna()
            c_tot = Counter(dict(now_df[now_df.columns[1]].value_counts()))
            n = now_df.shape[0]
            self.recursive_discrete(now_df, 0, n, c_tot, self.boundaries[k])

    def transform(self, continuous_data):
        col_list = []
        for k in continuous_data:
            if len(self.boundaries[k]) > 0:
                discrete_col = pd.DataFrame(pd.np.digitize(continuous_data[k], bins=self.boundaries[k], right=True),
                                            columns=[k])
                name_dict = {}
                for i in range(len(self.boundaries[k]) + 1):
                    if i == 0:
                        name_dict[i] = '<={}'.format(self.boundaries[k][i])
                    elif i == len(self.boundaries[k]):
                        name_dict[i] = '>{}'.format(self.boundaries[k][i - 1])
                    else:
                        name_dict[i] = '({}, {}]'.format(self.boundaries[k][i - 1], self.boundaries[k][i])
                discrete_col.replace(name_dict, inplace=True)
            else:
                discrete_col = pd.DataFrame(['ALL_RANGE'] * continuous_data.shape[0], columns=[k])
            col_list.append(discrete_col)
        return pd.concat(col_list, axis=1)

    @staticmethod
    def entropy(n, c_tot):
        if n <= 0:
            return 0.0
        e = 0.0
        for k, v in c_tot.items():
            p = v / n
            e += -p * math.log2(p) if p > 0.0 else 0.0
        return e

    @staticmethod
    def get_k(cnt):
        return len([v for v in cnt.values() if v > 0])

    def recursive_discrete(self, df, l, r, c_tot, boundaries):
        """Search the partition boundaries for each feature recursively."""
        n = r - l + 1
        e = self.entropy(n, c_tot)
        c_cnt1 = Counter()
        max_gain = 0.0
        max_pos = -1
        e1_keep = 0.0
        e2_keep = 0.0
        c_cnt1_keep = Counter()
        c_cnt2_keep = Counter()
        for i in range(l, r):
            row = df.iloc[i]
            c_cnt1[row[1]] += 1
            if i + 1 < r and row[0] == df.iloc[i + 1][0]:
                continue
            n1 = i - l + 1
            n2 = n - n1
            c_cnt2 = Counter()
            for k in c_tot:
                c_cnt2[k] = c_tot[k] - c_cnt1[k]
            e1 = self.entropy(n1, c_cnt1)
            e2 = self.entropy(n2, c_cnt2)
            ent_gain = e - (n1 / n) * e1 - (n2 / n) * e2
            if ent_gain > max_gain:
                max_gain = ent_gain
                max_pos = i
                e1_keep = e1
                e2_keep = e2
                c_cnt1_keep = c_cnt1.copy()
                c_cnt2_keep = c_cnt2.copy()

        k = self.get_k(c_tot)
        k1 = self.get_k(c_cnt1_keep)
        k2 = self.get_k(c_cnt2_keep)
        delta = math.log2(3 ** k - 2) - (k * e - k1 * e1_keep - k2 * e2_keep)
        gain_threshold = math.log2(n - 1) / n + delta / n

        if max_gain < gain_threshold or max_pos == -1:
            return
        if l < max_pos:
            self.recursive_discrete(df, l, max_pos + 1, c_cnt1_keep, boundaries)
        boundaries.append(df.iloc[max_pos][0])
        if max_pos + 1 < r - 1:
            self.recursive_discrete(df, max_pos + 1, r, c_cnt2_keep, boundaries)
