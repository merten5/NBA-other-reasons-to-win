import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from unidecode import unidecode
import os
import time
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import ast

def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Clean player names by removing '*'
    if 'Player' in df.columns:
        df['Player'] = df['Player'].str.replace('*', '', regex=False)
    
    # Transliterate special characters to their closest English equivalents
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].map(lambda x: unidecode(x) if isinstance(x, str) else x)
    
    return df

def pca_code(data, title, labels):
    # Handle missing values
    data.fillna(0, inplace=True)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=2)
    data_projected = pca.fit_transform(data_scaled)
    
    # DataFrames for eigenvalues, eigenvectors, and explained variance
    eigenvalues_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_))],
        'Eigenvalue': pca.explained_variance_, 
        'Explained Variance Ratio': pca.explained_variance_ratio_, 
        'Cumulative Explained Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    
    loadings_df = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=data.columns)
    
    projected_df = pd.DataFrame(data_projected, columns=['PC1', 'PC2'])
    projected_df['Label'] = labels

    return projected_df, eigenvalues_df, loadings_df

def compute_bic(kmeans, data):
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    n_clusters = centers.shape[0]
    n_samples = data.shape[0]
    n_features = data.shape[1]
    
    # Compute the within-cluster sum of squares
    wcss = np.sum([np.sum((data[labels == i] - centers[i]) ** 2) for i in range(n_clusters)])
    
    # Compute the log likelihood
    log_likelihood = -0.5 * wcss
    log_likelihood -= 0.5 * n_samples * n_features * np.log(2 * np.pi)
    log_likelihood -= 0.5 * n_samples * np.log(n_samples)
    
    # Compute the number of parameters
    n_params = n_clusters * (n_features + 1)
    
    # Compute the BIC
    bic = log_likelihood - 0.5 * n_params * np.log(n_samples)
    
    return bic

def find_optimal_k(data, max_k=10):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    bics = []
    for k in range(1, max_k + 1):
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=23, n_init=10)
        kmeans.fit(data_scaled)
        bic = compute_bic(kmeans, data_scaled)
        bics.append(bic)
        end_time = time.time()
        
        print(f"Processed k={k} in {end_time - start_time:.2f} seconds.")
    
    optimal_k = np.argmax(bics) + 1
    return optimal_k, bics

# Measure start time
start_time = time.time()

# Load the combined dataset
file_path = 'combined_nba_data_f.csv'
combined_data = load_and_preprocess_data(file_path)

# Extract the labels
labels = combined_data.iloc[:, 0]

# Select numeric data only and exclude "Age," "Salary," and "Year" based features
exclude_features = ['Age', 'Salary'] + [col for col in combined_data.columns if 'Year' in col]
numeric_data = combined_data.select_dtypes(include=[np.number])
numeric_data = numeric_data.drop(columns=[col for col in exclude_features if col in numeric_data.columns])

# Perform PCA
projected_df, eigenvalues_df, loadings_df = pca_code(numeric_data, title="PCA of Combined NBA Data", labels=labels)

# Print explained variance and cumulative variance along with principal component names
print("Explained Variance Ratio by each Principal Component:")
print(eigenvalues_df[['Principal Component', 'Eigenvalue', 'Explained Variance Ratio', 'Cumulative Explained Variance']])

# Print loadings
print("Principal Component Loadings:")
print(loadings_df)

# Export explained variance to CSV
eigenvalues_df.to_csv('pca_explained_variance_combo_data.csv', index=False)

# Export loadings to CSV
loadings_df.to_csv('pca_loadings_combo_data.csv', index=True)

# Find the optimal number of clusters using BIC
optimal_k, bics = find_optimal_k(numeric_data, max_k=30)
print(f"Optimal number of clusters: {optimal_k}")

# Plot BIC values
plt.plot(range(1, 31), bics, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('BIC')
plt.title('BIC for different numbers of clusters')
plt.savefig('bic_plot.png')

# Choosing 20 as that is where the BIC plot slows down.
optimal_k = 20

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(numeric_data)

# Perform k-means clustering with the optimal number of clusters
print(f"Starting KMeans clustering with k={optimal_k}")
kmeans_start_time = time.time()
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=23, n_init=10, max_iter=300, verbose=1)
kmeans.fit(normalized_data)
kmeans_end_time = time.time()

print(f"KMeans clustering completed in {kmeans_end_time - kmeans_start_time:.2f} seconds.")

combined_data['Cluster'] = kmeans.labels_

# Check the first few rows to ensure clusters were added correctly
print("Cluster assignment example:")
print(combined_data[['Player', 'Cluster']].head(25))

# Select a subset of features for the pair plot
selected_features = ['Ht_in', 'Wt', 'WS', 'MP_per_game', '2P_per_game', '3P_per_game', 'PTS', 'TRB%',
                     'ORB%', 'AST']
sample_data = combined_data[selected_features + ['Cluster']]

# Create output directory if it doesn't exist
output_dir = 'distribution_plots'
os.makedirs(output_dir, exist_ok=True)

# Plot distribution of clusters across original features
for column in numeric_data.columns:
    safe_column = column.replace('/', '_').replace('%', 'pct')
    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_data, x=column, hue='Cluster', element='step', stat='density', common_norm=False, palette='viridis', alpha=0.6)
    plt.title(f'Distribution of {column} by Cluster')
    plt.savefig(f'distribution_{safe_column}_by_cluster.png')
    plt.show()
    print(f"Distribution plot for {column} saved as 'distribution_{safe_column}_by_cluster.png'.")

# Save the clustered data
output_file_clustered = 'combined_nba_data_clustered.csv'
print(f"Saving clustered data to '{output_file_clustered}'...")
combined_data.to_csv(output_file_clustered, index=False)
print(f"Clustered data has been saved to '{output_file_clustered}'")

# Time pre pair plot
curr_time = time.time()
print(f"Total pre pair plot time: {curr_time - start_time:.2f} seconds")

# Analyze and visualize the clusters using pair plot
print("Creating pair plot...")
sns.pairplot(sample_data, hue='Cluster', palette='viridis', diag_kind='kde', plot_kws={'s': 1})
plt.suptitle('KMeans Clusters', y=1.02)
plt.savefig('kmeans_clusters_final.png')
print("Pair plot saved as 'kmeans_clusters_final.png'.")

# Time post pair plot
curr_time = time.time()
print(f"Total post pair plot time: {curr_time - start_time:.2f} seconds")

# Examine cluster centers
print("Cluster Centers:")
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_data.columns)
print(cluster_centers)
cluster_centers.to_csv('cluster_centers.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a heatmap to visualize cluster centers
plt.figure(figsize=(16, 10))
sns.heatmap(cluster_centers, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Heatmap of Cluster Centers')
plt.savefig('cluster_centers_heatmap.png')
plt.show()

# Measure end time
end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")

#################
#################
#################

import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    if 'Player' in df.columns:
        df['Player'] = df['Player'].str.replace('*', '', regex=False)
    return df

def find_matching_cluster(row, data):
    pk = row['Pk']
    ht_in = row['Ht_in']
    wt = round(row['Wt'], -1)
    matching_rows = data[(data['Exp'] == 0) & 
                         (data['Pk'].between(pk - 1, pk + 1) if pk != 1 else data['Pk'] == pk + 1) &
                         (data['Ht_in'].between(ht_in - 2, ht_in + 2)) &
                         (data['Wt'].between(wt - 11, wt + 11))]
    if not matching_rows.empty:
        return matching_rows['Cluster'].mode()[0]
    
    # If no exact matches, choose the mode of all rookies with 'Pk' plus or minus one
    fallback_rows = data[(data['Year'] == 0) & 
                         (data['Pk'].between(pk - 1, pk + 1) if pk != 1 else data['Pk'] == pk + 1)]
    
    if not fallback_rows.empty:
        return fallback_rows['Cluster'].mode()[0]
    
    return data['Cluster'].mode()[0]

def calculate_playoff_points(playoffs):
    if playoffs == "Won Finals":
        return 15
    elif playoffs == "Lost Finals":
        return 10
    elif playoffs in ["Lost W. Conf. Finals", "Lost E. Conf. Finals"]:
        return 6
    elif playoffs in ["Lost W. Conf. Semis", "Lost E. Conf. Semis"]:
        return 3
    elif playoffs in ["Lost W. Conf. 1st Rnd.", "Lost E. Conf. 1st Rnd."]:
        return 1
    else:
        return 0

def assign_cluster(row, previous_year_clusters, data):
    player = row['Player']
    previous_year = row['Year'] - 1
    exp = row['Exp']
    if exp == 0:
        return find_matching_cluster(row, data)
    else:
        for year in range(previous_year, 1979, -1):
            if (player, year) in previous_year_clusters:
                return previous_year_clusters[(player, year)]
    return np.nan

# Load the dataset
file_path = 'combined_nba_data_clustered.csv'  # Replace with the actual file path
data = load_and_preprocess_data(file_path)

# Convert 'Year' to integer
data['Year'] = data['Year'].astype(int)

# Calculate playoff points
data['Playoff_Success'] = data['Playoffs'].apply(calculate_playoff_points)

# Build a dictionary to hold previous year clusters
previous_year_clusters = {(row['Player'], row['Year']): row['Cluster'] for index, row in data.iterrows()}

# Apply the function to assign clusters
data['Previous_Year_Cluster'] = data.apply(assign_cluster, axis=1, previous_year_clusters=previous_year_clusters, data=data)
data['Previous_Year_Cluster'].fillna(20.0, inplace=True)

# Calculate the number of teams each player played for during the year
player_team_counts = data.groupby(['Player', 'Year'])['Team'].nunique().reset_index()
player_team_counts.rename(columns={'Team': 'Num_Teams'}, inplace=True)

# Merge the number of teams back into the main dataframe
data = pd.merge(data, player_team_counts, on=['Player', 'Year'])

# Determine if the player played with the team the year before, is a rookie, or is not a rookie and did not play with the team the year before
data['Player_Status'] = data.apply(lambda row: 'Rookie' if row['Exp'] == 0 else
                                   ('With team last year' if (row['Player'], row['Year'] - 1) in previous_year_clusters and previous_year_clusters[(row['Player'], row['Year'] - 1)] == row['Cluster'] else
                                    'NOT with team last year'), axis=1)

# Remove data for the year 1980
data = data[data['Year'] > 1980]

data['Cluster'] = data['Cluster'].astype(float)

# Create the final dataframe with additional information
final_data = data.groupby(['Team', 'Year']).agg({
    'Playoff_Success': 'first',
    'Previous_Year_Cluster': lambda x: list(x),
    'Cluster': lambda x: list(x),
    'Coaches': 'first',
    'salary': lambda x: list(x),
    'Num_Teams': lambda x: list(x),
    'Player_Status': lambda x: list(x),
    'Age': lambda x: list(x),
    'Player': lambda x: list(x)
}).reset_index()

# Display the final dataframe
print(final_data.head())

# Save the final dataframe to a CSV file
final_data.to_csv('final_nba_data_with_players.csv', index=False)

#######
#######
#######

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load the processed final data
file_path = 'final_nba_data_with_players.csv'
final_data = pd.read_csv(file_path)

# Prepare the feature matrix and target vector
final_data['Previous_Year_Cluster'] = final_data['Previous_Year_Cluster'].apply(eval)

# Find the maximum length of the lists
max_len = final_data['Previous_Year_Cluster'].apply(len).max()

# Pad lists to the maximum length
X = final_data['Previous_Year_Cluster'].apply(lambda x: x + [20.0] * (max_len - len(x)))
X = np.array(X.tolist())
y = final_data['Playoff_Success']

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Standardize the feature matrix
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize and train the K-NN model with 6 neighbors since there are 6 choices
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Save the predictions to a CSV file for further analysis if needed
predictions = pd.DataFrame({
    'Team': final_data.loc[y_test.index, 'Team'],
    'Year': final_data.loc[y_test.index, 'Year'],
    'Actual_Playoff_Success': y_test,
    'Predicted_Playoff_Success': y_pred
})
predictions.to_csv('knn_playoff_success_predictions_base.csv', index=False)

print(predictions.head())

# Calculate permutation importance
result = permutation_importance(knn, x_test, y_test, n_repeats=10, n_jobs=-1)

# Get feature names
feature_names = ['Previous_Year_Cluster_' + str(i) for i in range(max_len)]

# Create a description for each feature (You can customize this mapping)
feature_descriptions = {
    f'Previous_Year_Cluster_{i}': f'Cluster {i}' for i in range(max_len)
}

# Print permutation feature importance
importance_df = pd.DataFrame(result.importances_mean, index=feature_names, columns=['Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Select top 20 and bottom 10 features
top_20 = importance_df.head(20)
bottom_10 = importance_df.tail(10)
importance_df_top_bottom = pd.concat([top_20, bottom_10])

# Save the feature importance to a CSV file
importance_df.to_csv('knn_base_feature_importance.csv')

# Plotting top and bottom feature importance
plt.figure(figsize=(10, 8))
importance_df_top_bottom.plot(kind='barh', legend=False)

# Plotting top and bottom feature importance
plt.figure(figsize=(10, 8))
importance_df_top_bottom.plot(kind='barh', legend=False)
plt.title('Feature Importance')
plt.xlabel('Mean Decrease in Accuracy')
plt.ylabel('Features')
plt.gca().invert_yaxis()
plt.yticks(fontsize=6)

# Save the plot as an image file
plt.savefig('feature_importance_plot_base_abs.png')

# Add a description column to the DataFrame
importance_df['Description'] = importance_df.index.map(feature_descriptions)

# Save the feature importance to a CSV file
importance_df.to_csv('knn_base_feature_importance_with_descriptions.csv')

#########
#### More predictions : Salary
#########

# Load the processed final data
file_path = 'final_nba_data_with_players.csv'
final_data = pd.read_csv(file_path)

# Filter data for years >= 1985
final_data = final_data[final_data['Year'] >= 1985]

# Replace 'nan' strings with None in the salary column before evaluation
final_data['salary'] = final_data['salary'].str.replace('nan', 'None')

# Convert string representations of lists to actual lists
final_data['Previous_Year_Cluster'] = final_data['Previous_Year_Cluster'].apply(ast.literal_eval)
final_data['salary'] = final_data['salary'].apply(ast.literal_eval)

# Replace None in salary with the minimum value in their row (excluding None)
def replace_none_with_min(salary_list):
    min_value = min([s for s in salary_list if s is not None], default=0.0)
    return [s if s is not None else min_value for s in salary_list]

final_data['salary'] = final_data['salary'].apply(replace_none_with_min)

# Transform salary into proportions
def salary_to_proportion(salary_list):
    total_salary = sum(salary_list)
    return [s / total_salary for s in salary_list] if total_salary > 0 else salary_list

final_data['salary'] = final_data['salary'].apply(salary_to_proportion)

# Find the maximum length of the lists
max_len_cluster = final_data['Previous_Year_Cluster'].apply(len).max()
max_len_salary = final_data['salary'].apply(len).max()

# Pad lists to the maximum length
previous_year_cluster_padded = final_data['Previous_Year_Cluster'].apply(lambda x: x + [20.0] * (max_len_cluster - len(x)))
salary_padded = final_data['salary'].apply(lambda x: x + [0.0] * (max_len_salary - len(x)))

# Convert to numpy arrays
x_cluster = np.array(previous_year_cluster_padded.tolist())
x_salary = np.array(salary_padded.tolist())

# Normalize the salary data
scaler_salary = StandardScaler()
x_salary = scaler_salary.fit_transform(x_salary)

# Combine the features
X = np.hstack((x_cluster, x_salary))
y = final_data['Playoff_Success']

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Standardize the combined feature matrix (excluding salary normalization done earlier)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize and train the K-NN model with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Save the predictions to a CSV file for further analysis if needed
predictions = pd.DataFrame({
    'Team': final_data.loc[y_test.index, 'Team'],
    'Year': final_data.loc[y_test.index, 'Year'],
    'Actual_Playoff_Success': y_test,
    'Predicted_Playoff_Success': y_pred
})
predictions.to_csv('knn_playoff_success_predictions_salary.csv', index=False)

print(predictions.head())

#########
#### More predictions : age
#########

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import ast

# Load the processed final data
file_path = 'final_nba_data_with_players.csv'
final_data = pd.read_csv(file_path)

# Replace 'nan' strings with None in the Age column before evaluation
final_data['Age'] = final_data['Age'].str.replace('nan', 'None')

# Convert string representations of lists to actual lists
final_data['Previous_Year_Cluster'] = final_data['Previous_Year_Cluster'].apply(ast.literal_eval)
final_data['Age'] = final_data['Age'].apply(ast.literal_eval)

# Function to replace None with the mean value in the list (excluding None)
def replace_none_with_mean(age_list):
    mean_value = np.mean([age for age in age_list if age is not None])
    return [age if age is not None else mean_value for age in age_list]

final_data['Age'] = final_data['Age'].apply(replace_none_with_mean)

# Find the maximum length of the lists
max_len_cluster = final_data['Previous_Year_Cluster'].apply(len).max()
max_len_age = final_data['Age'].apply(len).max()

# Pad lists to the maximum length
previous_year_cluster_padded = final_data['Previous_Year_Cluster'].apply(lambda x: x + [20.0] * (max_len_cluster - len(x)))
age_padded = final_data['Age'].apply(lambda x: x + [0.0] * (max_len_age - len(x)))

# Convert to numpy arrays
x_cluster = np.array(previous_year_cluster_padded.tolist())
x_age = np.array(age_padded.tolist())

# Normalize the age data
scaler_age = StandardScaler()
x_age = scaler_age.fit_transform(x_age)

# Combine the features
X = np.hstack((x_cluster, x_age))
y = final_data['Playoff_Success']

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Standardize the combined feature matrix
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize and train the K-NN model with 6 neighbors since there are 6 choices
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Save the predictions to a CSV file for further analysis if needed
predictions = pd.DataFrame({
    'Team': final_data.loc[y_test.index, 'Team'],
    'Year': final_data.loc[y_test.index, 'Year'],
    'Actual_Playoff_Success': y_test,
    'Predicted_Playoff_Success': y_pred
})
predictions.to_csv('knn_playoff_success_predictions_age.csv', index=False)

print(predictions.head())


#########
#### More predictions : Players added when and how
#########

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import ast

# Load the processed final data
file_path = 'final_nba_data_with_players.csv'
final_data = pd.read_csv(file_path)

# Convert string representations of lists to actual lists
final_data['Previous_Year_Cluster'] = final_data['Previous_Year_Cluster'].apply(ast.literal_eval)
final_data['Player_Status'] = final_data['Player_Status'].apply(ast.literal_eval)

# Flatten the Player_Status list and encode the strings as numerical factors
flat_player_status = [status for sublist in final_data['Player_Status'] for status in sublist]
le = LabelEncoder()
le.fit(flat_player_status)

# Apply the encoding to each list in the Player_Status column
final_data['Player_Status'] = final_data['Player_Status'].apply(lambda x: le.transform(x).tolist())

# Find the maximum length of the lists
max_len_cluster = final_data['Previous_Year_Cluster'].apply(len).max()
max_len_status = final_data['Player_Status'].apply(len).max()

# Pad lists to the maximum length
previous_year_cluster_padded = final_data['Previous_Year_Cluster'].apply(lambda x: x + [20.0] * (max_len_cluster - len(x)))
status_padded = final_data['Player_Status'].apply(lambda x: x + [0] * (max_len_status - len(x)))

# Convert to numpy arrays
x_cluster = np.array(previous_year_cluster_padded.tolist())
x_status = np.array(status_padded.tolist())

# Combine the features
X = np.hstack((x_cluster, x_status))
y = final_data['Playoff_Success']

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Standardize the combined feature matrix
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize and train the K-NN model with 6 neighbors since there are 6 choices
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Save the predictions to a CSV file for further analysis if needed
predictions = pd.DataFrame({
    'Team': final_data.loc[y_test.index, 'Team'],
    'Year': final_data.loc[y_test.index, 'Year'],
    'Actual_Playoff_Success': y_test,
    'Predicted_Playoff_Success': y_pred
})
predictions.to_csv('knn_playoff_success_predictions_player_status.csv', index=False)

print(predictions.head())

#########
#### More predictions : Coach
#########

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Load the processed final data
file_path = 'final_nba_data_with_players.csv'
final_data = pd.read_csv(file_path)

# Prepare the feature matrix and target vector
final_data['Previous_Year_Cluster'] = final_data['Previous_Year_Cluster'].apply(eval)

# Extract coach names by removing the (##-##) parts
def extract_coach_names(coach_str):
    return ', '.join(re.findall(r'([A-Za-z\s.]+)\s*\(\d+-\d+\)', coach_str))

final_data['Coaches'] = final_data['Coaches'].apply(extract_coach_names)

# Split coach names into individual names
final_data['Coaches'] = final_data['Coaches'].str.split(', ')

# Create a set of all unique coaches
all_coaches = set(sum(final_data['Coaches'].tolist(), []))

# Create a DataFrame to hold the one-hot encoding of coaches
coach_df = pd.DataFrame(0, index=final_data.index, columns=list(all_coaches))

# Fill the DataFrame with 1s where the coach is present
def fill_coach_df(row):
    coach_df.loc[row.name, row['Coaches']] = 1

final_data.apply(fill_coach_df, axis=1)

# Drop the original 'Coaches' column as it's no longer needed
final_data = final_data.drop(columns=['Coaches'])

# Combine the coach_df with final_data
final_data = pd.concat([final_data, coach_df], axis=1)

# Prepare the Previous_Year_Cluster feature matrix
max_len = final_data['Previous_Year_Cluster'].apply(len).max()
X_cluster = final_data['Previous_Year_Cluster'].apply(lambda x: x + [20.0] * (max_len - len(x)))
X_cluster = np.array(X_cluster.tolist())

# Combine the cluster data with the coach one-hot encoding
X = np.hstack((X_cluster, coach_df.values))

# Prepare the target vector
y = final_data['Playoff_Success']

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Standardize the feature matrix
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize and train the K-NN model with 6 neighbors since there are 6 choices
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Save the predictions to a CSV file for further analysis if needed
predictions = pd.DataFrame({
    'Team': final_data.loc[y_test.index, 'Team'],
    'Year': final_data.loc[y_test.index, 'Year'],
    'Actual_Playoff_Success': y_test,
    'Predicted_Playoff_Success': y_pred
})
predictions.to_csv('knn_playoff_success_predictions_coaches.csv', index=False)

print(predictions.head())

# Calculate permutation importance
result = permutation_importance(knn, x_test, y_test, n_repeats=10, n_jobs=-1)

# Get feature names
feature_names = ['Previous_Year_Cluster_' + str(i) for i in range(max_len)] + list(all_coaches)

# Print permutation feature importance
importance_df = pd.DataFrame(result.importances_mean, index=feature_names, columns=['Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Select top 35 and bottom 15 features
top_25 = importance_df.head(35)
bottom_25 = importance_df.tail(15)
importance_df_top_bottom = pd.concat([top_25, bottom_25])

# Save the feature importance to a CSV file
importance_df.to_csv('knn_coach_feature_importance.csv')

# Plotting top and bottom in feature importance
plt.figure(figsize=(10, 8))
importance_df_top_bottom.plot(kind='barh', legend=False)
plt.title('Top and Bottom in Feature Importance')
plt.xlabel('Mean Decrease in Accuracy')
plt.ylabel('Features')
plt.gca().invert_yaxis()
plt.yticks(fontsize=6)

# Save the plot as an image file
plt.savefig('feature_importance_plot_coach_abs.png')
