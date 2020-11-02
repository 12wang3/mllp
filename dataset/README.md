## Data set description
We have already put 12 UCI data sets in the `dataset` folder. The description of these data sets are listed in the table below:

|Dataset (Links) | \#Instances | \#Classes	| \#Original Features	| \#Binary Features|
| --- | --- | --- | --- | --- |
|[adult](https://archive.ics.uci.edu/ml/datasets/Adult)| 32561 | 2 | 14 | 155|
|[bank-marketing](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing) | 45211 | 2 | 16 | 88|
|[banknote](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) | 1372 | 2 | 4 | 17|
|[blogger](https://archive.ics.uci.edu/ml/datasets/BLOGGER) | 100 | 2 | 5 | 15|
|[chess](https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King%29) | 28056 | 18 | 6 | 40|
|[connect-4](http://archive.ics.uci.edu/ml/datasets/connect-4) | 67557 | 3 | 42 | 126|
|[letRecog](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition) | 20000 | 26 | 16 | 155|
|[magic04](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope) | 19020 | 2 | 10 | 79|
|[mushroom](https://archive.ics.uci.edu/ml/datasets/mushroom) | 8124 | 2 | 22 | 117|
|[nursery](https://archive.ics.uci.edu/ml/datasets/nursery) | 12960 | 5 | 8 | 27|
|[tic-tac-toe](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame) | 958 | 2 | 9 | 27|
|[wine](https://archive.ics.uci.edu/ml/datasets/wine) | 178 | 3 | 13 | 37|
For more information about these data sets, you can click the corresponding links.

## Data set format
Each data set corresponds to two files, `*.data` and `*.info`. The `*.data` file stores the data for each instance. The `*.info` file stores the information for each feature.

#### *.data
One row in `*.data` corresponds to one instance and one column corresponds to one feature (including the class label).

For example, the `adult.data`:

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 37 | Private | 284582 | Masters | 14 | Married-civ-spouse | Exec-managerial | Wife | White | Female | 0 | 0 | 40 | United-States | <=50K |
| 49 | Private | 160187 | 9th | 5 | Married-spouse-absent | Other-service | Not-in-family | Black | Female | 0 | 0 | 16 | Jamaica | <=50K |
| 52 | Self-emp-not-inc | 209642 | HS-grad | 9 | Married-civ-spouse | Exec-managerial | Husband | White | Male | 0 | 0 | 45 | United-States | >50K |
| 31 | Private | 45781 | Masters | 14 | Never-married | Prof-specialty | Not-in-family | White | Female | 14084 | 0 | 50 | United-States | >50K |
| ......|

#### *.info
One row (except the last row) in `*.info` corresponds to one feature (including the class label). The order of these features must be the same as the columns in `*.data`. The first column is the feature name, and the second column indicates the characteristics of the feature, i.e., continuous or discrete. The last row does not correspond to one feature. It specifies the position of the class label column.

For example, the `adult.info`:

| | |
| --- | --- |
|age | continuous |
|workclass | discrete |
|fnlwgt | continuous |
|education | discrete |
|education-num | continuous |
|...... | |
|hours-per-week | continuous |
|native-country | discrete |
|class | discrete |
|LABEL_POS | -1 |

## Add a new data set
You can run the demo on your own data sets by putting corresponding `*.data` and `*.info` file into the `dataset` folder. The formats of these two files are described above.