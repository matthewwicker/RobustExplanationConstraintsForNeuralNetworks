import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_ADULT_TRAIN = './Datasets/adult.data.csv'
DATA_ADULT_TEST = './Datasets/adult.test.csv'
DATA_CRIME_FILENAME = './Datasets/crime.csv'
DATA_GERMAN_FILENAME = './Datasets/german.csv'

class TabularDataset:

    def __init__(self, X_raw, y_raw, sensitive_features=[], drop_columns=[], drop_first=False, drop_first_labels=True):
        """
        X_raw: features dataframe
        y_raw: labels dataframe or column label
        sensitive_features: the features considered sensitive
        drop_columns: the columns considered superfluous and to be deleted
        drop_first: whether to drop first when one-hot encoding features
        drop_first_labels: whether to drop first when one-hot encoding labels
        """

        self.sensitive_features = sensitive_features

        X_raw.drop(columns=drop_columns, inplace=True)
        self.X_raw = X_raw

        num_cols, cat_cols, sens_num_cols, sens_cat_cols = self.get_num_cat_columns_sorted(X_raw, sensitive_features)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.sens_num_cols = sens_num_cols
        self.sens_cat_cols = sens_cat_cols
        self.original_columns = X_raw.columns.values.tolist()

        X_df_all, y_df = self.prepare_dataset(X_raw, y_raw, drop_first=drop_first, drop_first_labels=drop_first_labels, drop_columns=drop_columns)

        self.y_df = y_df

        # Map from original column names to all new encoded ones
        all_columns_map = {}
        encoded_columns = X_df_all.columns.values.tolist()
        for c in self.original_columns:
            all_columns_map[c] = [ encoded_columns.index(e_c) for e_c in encoded_columns if e_c == c or e_c.startswith(c + '_') ]

        # List of list of the indexes of each sensitive features
        encoded_features = X_df_all.columns.values.tolist()
        sensitive_idxs = []
        sensitive_idxs_flat = []
        for sf in sensitive_features:
            sensitive_idxs.append(all_columns_map[sf])
            sensitive_idxs_flat.extend(all_columns_map[sf])
        all_idxs = [i for i in range(len(X_df_all.columns))]
        valid_idxs = [i for i in all_idxs if i not in sensitive_idxs_flat]

        # Datasets with one-hot encoded columns of each sensitive feature
        self.sensitive_dfs = [X_df_all.iloc[:, idxs] for idxs in sensitive_idxs]

        # Dataset with all features but the sensitive ones
        self.X_df = X_df_all.iloc[:, valid_idxs]

        self.columns_map = {}
        encoded_columns = self.X_df.columns.values.tolist()
        for c in num_cols + cat_cols:
            self.columns_map[c] = [ encoded_columns.index(e_c) for e_c in encoded_columns if e_c == c or e_c.startswith(c + '_') ]

    def get_num_cat_columns_sorted(self, X_df, sensitive_features):
        num_cols = []
        cat_cols = []

        sens_num_cols = []
        sens_cat_cols = []

        for c in X_df.columns:
            if c in sensitive_features:
                if X_df[c].dtype == 'object' or X_df[c].dtype.name == 'category':
                    sens_cat_cols.append(c)
                else:
                    sens_num_cols.append(c)
            else:
                if X_df[c].dtype == 'object' or X_df[c].dtype.name == 'category':
                    cat_cols.append(c)
                else:
                    num_cols.append(c)

        num_cols.sort()
        cat_cols.sort()
        sens_num_cols.sort()
        sens_cat_cols.sort()

        return num_cols, cat_cols, sens_num_cols, sens_cat_cols

    def scale_num_cols(self, X_df, num_cols):
        """
        X_df: features dataframe
        num_cols: name of all numerical columns to be scaled
        returns: feature dataframe with scaled numerical features
        """
        X_df_scaled = X_df.copy()
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(X_df_scaled[num_cols])

        for i, c in enumerate(num_cols):
            X_df_scaled[c] = X_num[:,i]

        return X_df_scaled

    def process_num_cat_columns(self, X_df, drop_first):
        """
        X_df: features dataframe
        returns: feature dataframe with scaled numerical features and one-hot encoded categorical features
        """
        num_cols = []
        cat_cols = []

        for c in X_df.columns:
            if X_df[c].dtype == 'object' or X_df[c].dtype.name == 'category':
                cat_cols.append(c)
            else:
                num_cols.append(c)

        # TODO: need to think about this drop_first
        X_df_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=drop_first)

        cat_cols = list(set(X_df_encoded.columns) - set(num_cols))

        num_cols.sort()
        cat_cols.sort()

        X_df_encoded_scaled = self.scale_num_cols(X_df_encoded, num_cols)

        return X_df_encoded_scaled[num_cols + cat_cols]


    def process_labels(self, X_df, y_df, drop_first):
        X_processed = X_df.copy()
        if isinstance(y_df, str):
            prefix = y_df
            y_columns = [ c for c in X_processed.columns if c == prefix or c.startswith(prefix + '_') ]
            y_processed = X_df[y_columns]
            X_processed.drop(columns=y_columns, inplace=True)
        else:
            y_processed = pd.get_dummies(y_df, drop_first=drop_first)

        return X_processed, y_processed


    def prepare_dataset(self, X_df_original, y_df_original, drop_first, drop_first_labels, drop_columns=[]):
        """
        X_df_original: features dataframe
        y_df_original: labels dataframe
        returns:
            - feature dataframe with scaled numerical features and one-hot encoded categorical features
            - one hot encoded labels, with drop_first option
        """
        X_df = X_df_original.copy()

        X_processed = self.process_num_cat_columns(X_df, drop_first)

        X_processed, y_processed = self.process_labels(X_processed, y_df_original, drop_first_labels)

        return X_processed, y_processed
    
# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov,
#     Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th,
#     7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
#     Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
#     Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving,
#     Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany,
#     Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras,
#     Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France,
#     Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,
#     Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
def get_adult_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    """
    train_path: path to training data
    test_path: path to test data
    returns: tuple of training features, training labels, test features and test labels
    """
    train_df = pd.read_csv(DATA_ADULT_TRAIN, na_values='?').dropna()
    test_df = pd.read_csv(DATA_ADULT_TEST, na_values='?').dropna()
    target = 'target'

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[[target]]

    train_ds = TabularDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = TabularDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds



# CREDIT DATASET:
# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response
# variable. This study reviewed the literature and used the following 23 variables as explanatory
# variables:
#     x1: Amount of the given credit (NT dollar): it includes both the individual consumer
#         credit and his/her family (supplementary) credit.
#     x2: Gender (1 = male; 2 = female).
#     x3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
#     x4: Marital status (1 = married; 2 = single; 3 = others).
#     x5: Age (year).
#     x6 - x11: History of past payment. We tracked the past monthly payment records (from April to
#         September, 2005) as follows: x6 = the repayment status in September, 2005; x7 = the
#         repayment status in August, 2005; . . .;x11 = the repayment status in April, 2005. The
#         measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one
#         month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months;
#         9 = payment delay for nine months and above.
#     x12-x17: Amount of bill statement (NT dollar). x12 = amount of bill statement in September,
#         2005; x13 = amount of bill statement in August, 2005; . . .; x17 = amount of bill
#         statement in April, 2005.
#     x18-x23: Amount of previous payment (NT dollar). x18 = amount paid in September, 2005;
#         x19 = amount paid in August, 2005; . . .;x23 = amount paid in April, 2005.
def get_credit_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    """
    sensitive_features: features that should be considered sensitive when building the
        BiasedDataset object
    drop_columns: columns we can ignore and drop
    random_state: to pass to train_test_split
    return: two BiasedDataset objects, for training and test data respectively
    """
    credit_data = fetch_openml(data_id=42477, as_frame=True, data_home='./data/raw')

    # Force categorical data do be dtype: category
    features = credit_data.data
    categorical_features = ['x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
    for cf in categorical_features:
        features[cf] = features[cf].astype(str).astype('category')

    # Encode output
    target = (credit_data.target == "1") * 1
    target = pd.DataFrame({'target': target})

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state)

    train_ds = TabularDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = TabularDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds



def get_crime_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    data_df = pd.read_csv(DATA_CRIME_FILENAME, na_values='?').dropna()
    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
    target = 'ViolentCrimesPerPop'

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[[target]]

    train_ds = TabularDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = TabularDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds


def get_german_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    data_df = pd.read_csv(DATA_GERMAN_FILENAME, na_values='?').dropna()

    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
    target = 'target'

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[[target]]

    train_ds = TabularDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = TabularDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds
    
    