#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 23:13:04 2024

@author: manushipatel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
insurance_data = pd.read_csv('/Users/manushipatel/Documents/mock insurance policy project/insurance_dataset.csv')
synthetic_data = pd.read_csv('/Users/manushipatel/Documents/mock insurance policy project/data_synthetic.csv')

# Display the first few rows of each dataset
print("Insurance Dataset:")
insurance_data.head()

print("\nSynthetic Dataset:")
synthetic_data.head()

# Check for missing values in the datasets
missing_values_insurance = insurance_data.isnull().sum()
missing_values_synthetic = synthetic_data.isnull().sum()

print("Missing values in Insurance Dataset:")
print(missing_values_insurance)
print("\nMissing values in Synthetic Dataset:")
print(missing_values_synthetic)

# Handling missing values
# Insurance Data
for column in ['Age', 'Income', 'Claim_Amount']:
    median_value = insurance_data[column].median()
    insurance_data[column].fillna(median_value, inplace=True)

for column in ['Gender', 'Marital_Status', 'Education', 'Occupation']:
    mode_value = insurance_data[column].mode()[0]
    insurance_data[column].fillna(mode_value, inplace=True)
    
# Synthetic Data
numeric_columns_synthetic = ['Customer ID', 'Age', 'Income Level', 'Location', 'Claim History', 'Coverage Amount', 
                             'Premium Amount', 'Deductible', 'Risk Profile', 'Previous Claims History', 'Credit Score']
category_columns_synthetic = ['Gender', 'Marital Status', 'Occupation', 'Education Level', 'Geographic Information', 
                              'Behavioral Data', 'Purchase History', 'Policy Start Date', 'Policy Renewal Date', 
                              'Interactions with Customer Service', 'Insurance Products Owned', 'Policy Type', 
                              'Customer Preferences', 'Preferred Communication Channel', 'Preferred Contact Time', 
                              'Preferred Language', 'Driving Record', 'Life Events', 'Segmentation Group']

for column in numeric_columns_synthetic:
    median_value = synthetic_data[column].median()
    synthetic_data[column].fillna(median_value, inplace=True)

for column in category_columns_synthetic:
    mode_value = synthetic_data[column].mode()[0]
    synthetic_data[column].fillna(mode_value, inplace=True)
    
# Display the updated missing values count to confirm all are handled
print("\nUpdated Missing values in Insurance Dataset:")
print(insurance_data.isnull().sum())
print("\nUpdated Missing values in Synthetic Dataset:")
print(synthetic_data.isnull().sum())

# Summarizing key statistics for each variable in the insurance dataset
insurance_stats = insurance_data.describe()
insurance_stats.loc['range'] = insurance_stats.loc['max'] - insurance_stats.loc['min']
print("Key Statistics for Insurance Dataset:")
insurance_stats

# Summarizing key statistics for each variable in the synthetic dataset
synthetic_stats = synthetic_data.describe()
synthetic_stats.loc['range'] = synthetic_stats.loc['max'] - synthetic_stats.loc['min']
print("Key Statistics for Synthetic Dataset:")
synthetic_stats

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Histogram of Claim Amounts
plt.figure(figsize=(10, 6))
sns.histplot(insurance_data['Claim_Amount'], kde=True, color='blue')
plt.title('Distribution of Claim Amounts')
plt.xlabel('Claim Amount')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Claim Amounts
plt.figure(figsize=(10, 6))
sns.boxplot(x=insurance_data['Claim_Amount'], color='green')
plt.title('Box Plot of Claim Amounts')
plt.xlabel('Claim Amount')
plt.show()

# Selecting numeric and categorical columns
numeric_columns = insurance_data.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = insurance_data.select_dtypes(exclude=[np.number]).columns.tolist()

# Correlation matrix for numeric columns
plt.figure(figsize=(10, 8))
sns.heatmap(insurance_data[numeric_columns].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Variables')
plt.show()

# Scatter plots for numeric variables against Claim Amount
for column in numeric_columns:
    if column != 'Claim_Amount':
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=insurance_data, x=column, y='Claim_Amount', alpha=0.6)
        plt.title(f'Relationship between Claim Amount and {column}')
        plt.xlabel(column)
        plt.ylabel('Claim Amount')
        plt.show()
        
# Group comparisons for categorical variables
for column in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=column, y='Claim_Amount', data=insurance_data)
    plt.title(f'Claim Amount Distribution by {column}')
    plt.xlabel(column)
    plt.ylabel('Claim Amount')
    plt.xticks(rotation=45)
    plt.show()
    
# Analyzing the distribution of claim amounts for different demographic groups
# and identifying groups with higher or lower average claim amounts.

# Selecting categorical columns for analysis
categorical_columns = insurance_data.select_dtypes(exclude=[np.number]).columns.tolist()

# Creating a figure to plot multiple subplots
fig, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(10, 6 * len(categorical_columns)))

# Looping through each categorical column to plot the average claim amount for each category
for i, column in enumerate(categorical_columns):
    # Grouping data by categorical column and calculating mean claim amount
    group_data = insurance_data.groupby(column)['Claim_Amount'].mean().sort_values()
    
    # Creating a bar plot for each categorical column
    axes[i].bar(group_data.index, group_data.values, color='skyblue')
    axes[i].set_title(f'Average Claim Amount by {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Average Claim Amount')
    axes[i].tick_params(axis='x', rotation=45)

# Adjusting layout to prevent overlap
plt.tight_layout()
plt.show()

# Analyzing the distribution of policyholders across different demographic groups
# Selecting categorical columns for analysis
categorical_columns = insurance_data.select_dtypes(exclude=[np.number]).columns.tolist()

# Plotting the distribution of policyholders across different demographic groups
fig, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(10, 6 * len(categorical_columns)))
for i, column in enumerate(categorical_columns):
    sns.countplot(x=column, data=insurance_data, ax=axes[i], palette='Set2')
    axes[i].set_title(f'Distribution of Policyholders by {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Number of Policyholders')
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Identifying potential areas for policy optimization based on risk profiles and claim patterns
# Calculating average claim amounts and claim rates for each demographic group
grouped_data = insurance_data.groupby(categorical_columns)['Claim_Amount'].agg(['mean', 'count']).reset_index()
high_risk_groups = grouped_data[grouped_data['mean'] > grouped_data['mean'].quantile(0.75)]
low_risk_groups = grouped_data[grouped_data['mean'] < grouped_data['mean'].quantile(0.25)]

# Displaying high risk groups
print("High Risk Groups (Top 25% of Average Claim Amount):")
high_risk_groups

# Displaying low risk groups
print("Low Risk Groups (Bottom 25% of Average Claim Amount):")
low_risk_groups

# Suggestions for policy optimization
# 1. Consider adjusting premiums or coverage terms for high risk groups to mitigate potential losses."
# 2. Offer incentives or discounts to low risk groups to maintain customer loyalty and attract similar profiles."
