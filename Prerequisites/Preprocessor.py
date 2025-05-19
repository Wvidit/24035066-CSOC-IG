import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import RobustScaler
import scipy.stats as stats

def distribution(df):
    plt.figure(figsize=(15, 10))
    numeric_columns = df.select_dtypes(include=np.number).columns
    for i, col in enumerate(numeric_columns):
        plt.subplot(3, 3, i + 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()

def box_plot(df):
    df_melted = df.melt(var_name='Feature', value_name='Value')
    sns.boxplot(y='Feature', x='Value', data=df_melted)
    plt.title("Box Plots by Feature")
    plt.figure(figsize=(25, 6))
    plt.tight_layout()

def scale_df(df):
    columns = df.columns.str.capitalize()
    df.to_numpy()
    scaler = RobustScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    df = pd.DataFrame(df, columns = columns )
    print("Scaling Complete:")
    return df

def fitting(df):
    for i in df.columns:
        stats.probplot(df[i], dist="norm", plot=plt)
        plt.title("Distribution for " + i)
        plt.show()

def coerr_heatmap(df):
    sns.heatmap(df.corr(), annot=True)
