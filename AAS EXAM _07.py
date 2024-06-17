#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


# In[59]:


df=pd.read_csv("C:/Users/HP/Downloads/data - data (1).csv")


# In[60]:


df.head()


# In[61]:


df.tail()


# In[62]:


df.shape


# In[63]:


df.info()


# In[64]:


df.count()


# conclusion:with this we get to know how many rows are present in the column

# In[65]:


df.isnull().sum()


# CONCLUSION:
# this shows that there is no null values present in the DataFrame

# In[66]:


duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ",duplicate_rows_df.shape)


# conclusion: this shows that there are no duplicate present in the column 

# In[67]:


df.describe()


# #computing outliers

# In[ ]:


def impute_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Replace outliers with bounds
    df.loc[df[column] < lower_bound, column] = lower_bound
    df.loc[df[column] > upper_bound, column] = upper_bound

    return df

# List of columns to impute (excluding 'Bankrupt?')
columns_to_impute = df.columns.difference(['Bankrupt?','Current Liability to Current Assets','Net Income Flag'])

# Apply the imputation function to each column
for column in columns_to_impute:
    df = impute_outliers(df, column)

# Check the counts of 'Bankrupt?' after imputation
print(df['Bankrupt?'].value_counts())


# #removing outliers with the iqr

# In[69]:


for column in df.columns:
    plt.figure(figsize=(3, 3))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.show()


# plotting boxplot for after removing all the outliers

# In[70]:


# Compute the correlation matrix
correlation_matrix = df.corr()


# In[71]:


# Extract the correlation of each feature with the target variable
target_correlation = correlation_matrix['Bankrupt?'].sort_values(ascending=False)

# Plot the correlations with the target variable
plt.figure(figsize=(10, 20))
sns.heatmap(target_correlation.to_frame(), annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation with Target Variable (Bankrupt?)')
plt.show()


# GETTING CORRELATION AND SORTING THE CORRELATION

# In[74]:


target_correlation = correlation_matrix['Bankrupt?']
(target_correlation).sort_values(ascending=False) #printing correlation of each column with target column 'Bankrupt? '


# In[75]:


sorted_correlations = correlation_matrix['Bankrupt?'].abs().sort_values(ascending=False)
#used abs to convert negative values into postitve and sort the values in descending order


# In[86]:


df = df.drop('Liability-Assets Flag', axis=1)


# In[ ]:





# In[ ]:


df = df.drop('Net Income Flag', axis=1)


# In[82]:


df.shape


# In[83]:


df.describe()


# In[ ]:





# In[84]:


# Assuming 'df' is your DataFrame and 'Bankrupt?' is the column you want to plot
bankrupt_counts = df['Bankrupt?'].value_counts()

plt.figure(figsize=(5,5))
plt.pie(bankrupt_counts, labels=bankrupt_counts.index, autopct='%1.1f%%')
plt.show()


# #conclusion this shows that only 3.2 companies are bankrupt

# In[45]:


bankrupt_counts = df['Bankrupt?'].value_counts()

plt.figure(figsize=(5,5))
plt.pie(bankrupt_counts, labels=bankrupt_counts.index, autopct='%1.1f%%')
plt.show()


# #conclusion this shows that only 3.2 companies are bankrupt

# In[85]:


import pandas as pd
import matplotlib.pyplot as plt

# Plot histograms for all columns
df.hist(bins=50, figsize=(50,40))
plt.show()


# In[23]:


plt.figure(figsize=(50,40))
c=df.corr()
sns.heatmap(c,cmap="rocket",annot=True)
c
plt.show()


# #HYPOTHESIS

# In[21]:


from scipy.stats import ttest_ind

# Perform t-tests for each feature
t_test_results = {}
for feature in df:
    bankrupt = df[df['Bankrupt?'] == 1][feature]
    non_bankrupt = df[df['Bankrupt?'] == 0][feature]
    t_stat, p_value = ttest_ind(bankrupt, non_bankrupt, equal_var=False)  # Welch's t-test for unequal variances
    t_test_results[feature] = (t_stat, p_value)

# Convert the results to a DataFrame for better visualization
t_test_df = pd.DataFrame(t_test_results, index=['T-statistic', 'P-value']).T
t_test_df['Significant'] = t_test_df['P-value'] < 0.05
print(t_test_df)


# In[27]:


import scipy.stats as stats

# Perform t-tests for each feature
significant_features = []
for column in df.columns[:-1]:  # Exclude the target variable
    t_stat, p_value = stats.ttest_ind(df[column][df['Bankrupt?'] == 1], df[column][df['Bankrupt?'] == 0])
    if p_value < 0.05:
        significant_features.append(column)

# Display significant features
significant_features


# In[44]:


non_significant_rows = t_test_df[t_test_df['Significant'] == False]
non_significant_rows


# In[46]:


from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=10)
pca_features = pca.fit_transform(df.drop('Bankrupt?', axis=1))

# Create a DataFrame for PCA features
pca_df = pd.DataFrame(data=pca_features, columns=[f'PCA_{i+1}' for i in range(10)])
pca_df['Bankrupt?'] = df['Bankrupt?']

# Display the explained variance ratio
pca.explained_variance_ratio_


# In[47]:


import numpy as np

# Create interaction features
df['ROA(C)_times_Operating_Profit_Rate'] = df['ROA(C) before interest and depreciation before interest'] * df['Operating Profit Rate']
df['ROA(A)_times_Cash_Flow_Per_Share'] = df['ROA(A) before interest and % after tax'] * df['Cash Flow Per Share']

# Create ratio features
df['Operating_Profit_to_Expense_Rate'] = df['Operating Profit Rate'] / (df['Operating Expense Rate'] + 1e-5)  # Avoid division by zero

# Log transformations
df['Log_Operating_Profit_Per_Share'] = np.log1p(df['Operating Profit Per Share (Yuan ¥)'])
df['Log_Revenue_Per_Share'] = np.log1p(df['Revenue Per Share (Yuan ¥)'])

# Check the new features
print(df.head())


# In[48]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=['Bankrupt?']))

# Apply PCA
pca = PCA(n_components=10)  # Adjust the number of components as needed
pca_features = pca.fit_transform(scaled_features)

# Add PCA features to the dataframe
for i in range(pca_features.shape[1]):
    df[f'PCA_{i+1}'] = pca_features[:, i]

print(pca.explained_variance_ratio_)  # Print explained variance ratio to see the contribution of each component
print(df.head())


# In[49]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Define the model
model = LogisticRegression()

# Perform RFE
rfe = RFE(model, n_features_to_select=10)  # Select top 10 features
rfe = rfe.fit(df.drop(columns=['Bankrupt?']), df['Bankrupt?'])

# Get selected features
selected_features = df.drop(columns=['Bankrupt?']).columns[rfe.support_]
print("Selected features:", selected_features)

# Create a dataframe with only the selected features
df_selected = df[selected_features]
df_selected['Bankrupt?'] = df['Bankrupt?']
print(df_selected.head())


# In[50]:


from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df.drop('Bankrupt?', axis=1), df['Bankrupt?'], test_size=0.2, random_state=42)


# In[ ]:





# In[ ]:





# In[79]:


# Data preprocessing steps
df.dropna(inplace=True)

# # Detect and cap outliers using IQR
# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# IQR = Q3 - Q1
# df_capped = df.copy()
# for column in df.columns:
#     if column != 'Bankrupt?':
#         lower_bound = Q1[column] - 1.5 * IQR[column]
#         upper_bound = Q3[column] + 1.5 * IQR[column]
#         df_capped[column] = df[column].clip(lower_bound, upper_bound)

# Feature selection
model = LogisticRegression(max_iter=1000)
selected_features = df_capped.columns[:-1]  # Use all features

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df_capped[selected_features], df_capped['Bankrupt?'], test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = logistic_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print(cm)

# Optionally, visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logistic_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()


# Key Insights:
# Positive Correlations with Bankruptcy Probability:
# 
# Borrowing dependency (0.278), Total debt/Total net worth (0.273), Debt ratio % (0.247), and Liability to Equity (0.246) show moderate to strong positive correlations with the likelihood of bankruptcy. This suggests that companies with higher levels of borrowing relative to their assets or equity are more likely to face financial distress leading to bankruptcy.
# Negative Correlations:
# 
# Retained Earnings to Total Assets (-0.255), Net Income to Total Assets (-0.256), and Persistent EPS in the Last Four Seasons (-0.256) exhibit moderate negative correlations with bankruptcy probability. Companies with higher retained earnings, net income, and persistent earnings per share (EPS) relative to their total assets tend to have lower risks of bankruptcy. These metrics indicate stronger financial health and profitability.
# Financial Health Indicators:
# 
# Ratios related to profitability (e.g., net income), asset management (e.g., total assets), and financial leverage (e.g., debt ratios) are crucial in assessing bankruptcy risk. Companies should maintain healthy levels of profitability, manage debt effectively, and strengthen asset management practices to mitigate bankruptcy risk.
# Implications for Financial Risk Management:
# Early Warning Signs:
# 
# Monitoring borrowing dependency, debt ratios, and liability to equity ratios can serve as early warning signs of potential financial distress. Financial managers should regularly assess these ratios to identify companies at higher risk of bankruptcy.
# Strategic Financial Planning:
# 
# Emphasize strategies to reduce borrowing dependency and optimize debt levels relative to assets and equity. This includes refinancing debts at favorable terms, negotiating with creditors, and diversifying funding sources to enhance financial stability.
# Profitability and Efficiency:
# 
# Focus on improving profitability metrics such as net income and retained earnings. Enhancing operational efficiency and cost management can bolster financial resilience against economic downturns or market volatility.
# Scenario Analysis and Stress Testing:
# 
# Conduct scenario analyses to evaluate the impact of adverse economic conditions or market shocks on financial ratios. Stress testing helps identify vulnerabilities and prepares contingency plans to mitigate potential risks.
# Actionable Recommendations:
# Enhance Financial Reporting and Transparency:
# 
# Improve transparency in financial reporting to accurately reflect borrowing levels, debt structures, and profitability metrics. Clear and comprehensive financial disclosures facilitate informed decision-making and risk assessment.
# Implement Robust Risk Management Practices:
# 
# Develop and implement robust risk management frameworks that integrate quantitative analysis of financial ratios with qualitative assessments of market conditions and operational risks.
# Invest in Data Analytics and Predictive Modeling:
# 
# Leverage advanced data analytics and predictive modeling techniques to enhance bankruptcy prediction capabilities. Develop machine learning models that incorporate significant financial ratios identified through correlation analysis for accurate risk assessment.
# Continuous Monitoring and Evaluation:
# 
# Establish regular monitoring mechanisms to track changes in key financial ratios and update risk assessments accordingly. Continuous evaluation ensures proactive risk management and timely intervention.

# In[ ]:




