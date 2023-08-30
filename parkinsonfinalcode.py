import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Read txt into a DataFrame
df = pd.read_csv("po1_data.txt")

# inspect DataFrame content
print("Content of DataFrame: ")
print(df.info())
print(df.head())
print("")

# Splitting the dataframe
df1 = df[df["PDIndicator"] == 0]
df1.info() 
df2 = df[df["PDIndicator"] == 1]
df2.info()
#creating a list of all columns of the datasets but features start from 2nd index and end at 27th index
feature_columns = df.columns[2:27]
 
setofsalientfeatures = []

# Iterate over each feature
for feature in feature_columns:
    print("\n\n", feature)
    sample1 = np.array(df1[feature]) #subset of healthy dataset 
    sample2 = np.array(df2[feature])
    n1 = len(sample1)
    n2 = len(sample2)
    x_bar1 = st.tmean(sample1)
    s1 = st.tstd(sample1)
    x_bar2 = st.tmean(sample2)
    s2 = st.tstd(sample2)    
# Perform two-sample t-test #assuming simple random sampling and normal distribution of data
    t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1, x_bar2, s2, n2, equal_var=False, alternative='two-sided')
    print("\t t-statistic (t*): %.2f" % t_stats)
    print("\t p-value: %.4f" % p_val)
    print("\n Conclusion:")
    if p_val < 0.05: #confidence level 95%
        print("\t We reject the null hypothesis.")
        setofsalientfeatures.append(feature)  #append the features with rejected null hypothesis
    else:
        print("\t We accept the null hypothesis.")

print(setofsalientfeatures)    

# Visualization using histograms for each salient feature and grouped by PDIndicator
for feature in setofsalientfeatures:
    plt.figure(figsize=(8, 6))
    plt.hist(df1[feature], alpha=0.5, label='PDIndicator = 0', bins=20)
    plt.hist(df2[feature], alpha=0.5, label='PDIndicator = 1', bins=20)
    plt.title(f'Histogram of {feature} by PDIndicator')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
    
#Visualization using boxplot for each feature and grouped by PDIndicator on x axis
for feature in feature_columns:
    df.boxplot(column=feature, by='PDIndicator')
    plt.title(f'Boxplot of {feature} by PDIndicator')
    plt.ylabel(feature)
    plt.xlabel('PDIndicator')
    plt.show()
    

    
    
    
        

