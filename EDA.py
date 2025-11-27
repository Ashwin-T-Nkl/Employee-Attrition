import pandas as pd
import matplotlib.pyplot as plt

INPUT = "clean_employee.csv"

def load_data():
    df = pd.read_csv(INPUT)
    return df


def attrition_count(df):
    fig, ax = plt.subplots()
    df["Attrition"].value_counts().plot(kind='bar', ax=ax, color="#775C5C", edgecolor="black" )
    ax.set_title("Attrition Count")
    ax.set_xlabel("Attrition (0 = Stay, 1 = Leave)")
    ax.set_ylabel("Count")
    return fig



def age_distribution(df):
    fig, ax = plt.subplots()
    ax.hist(df["Age"], bins=15, edgecolor='black', color="#BD6CEC")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    return fig


def income_distribution(df):
    fig, ax = plt.subplots()
    ax.hist(df["MonthlyIncome"], bins=15, edgecolor='black', color='lightpink')
    ax.set_title("Monthly Income Distribution")
    ax.set_xlabel("Monthly Income")
    ax.set_ylabel("Frequency")
    return fig


def department_attrition(df):
    fig, ax = plt.subplots()
    df.groupby("Department")["Attrition"].mean().plot(kind='bar', ax=ax, color="#58B4D0")
    ax.set_title("Attrition Rate by Department")
    ax.set_xlabel("Department")
    ax.set_ylabel("Attrition Rate")
    return fig


def jobrole_attrition(df):
    fig, ax = plt.subplots()
    df.groupby("JobRole")["Attrition"].mean().plot(kind='bar', ax=ax, color="#D6B94E")
    ax.set_title("Attrition Rate by Job Role")
    ax.set_xlabel("Job Role")
    ax.set_ylabel("Attrition Rate")
    return fig
