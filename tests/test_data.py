import pytest
import pandas as pd

train_df = pd.read_csv('data/processed/train.csv')
eval_df = pd.read_csv('data/processed/eval.csv')

def test_columns_exist():
    expected = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    assert all(col in train_df.columns for col in expected)
    assert all(col in eval_df.columns for col in expected)

def test_no_missing_values():
    assert train_df.isnull().sum().sum() == 0
    assert eval_df.isnull().sum().sum() == 0

def test_feature_types():
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in feature_cols:
        assert train_df[col].dtype in ['float64', 'int64']
        assert eval_df[col].dtype in ['float64', 'int64']
    assert train_df['target'].dtype == 'int64'
    assert eval_df['target'].dtype == 'int64'

def test_value_ranges():
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in feature_cols:
        assert train_df[col].between(0, 10).all()
        assert eval_df[col].between(0, 10).all()
    assert train_df['target'].between(0, 2).all()
    assert eval_df['target'].between(0, 2).all()
