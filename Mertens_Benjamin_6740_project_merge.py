import pandas as pd
from unidecode import unidecode
import re

def load_data(file_path, encoding='ISO-8859-1'):
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"Loaded {file_path} with {len(df)} records and columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def clean_player_names(df, col_name='Player'):
    if col_name in df.columns:
        df[col_name] = df[col_name].str.replace('*', '', regex=False)
    return df

def remove_tot_teams(df, col_name='Team'):
    if col_name in df.columns:
        df = df[df[col_name] != 'TOT']
    return df

def rename_columns(df, column_map):
    df.rename(columns=column_map, inplace=True)
    return df

def convert_year_column(df, col_name='Year'):
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str)
    return df

def extract_year(season):
    if isinstance(season, str):
        match = re.search(r'(\d{4})-(\d{2})', season)
        if match:
            return match.group(1)
        else:
            return season
    return season

def extract_and_convert_year(df, season_col, year_col):
    if season_col in df.columns:
        df[year_col] = df[season_col].apply(extract_year)
    df.drop(columns=[season_col], inplace=True, errors='ignore')
    df = convert_year_column(df, year_col)
    return df

def calculate_playoff_success(data, start_year, end_year):
    # Initialize the columns
    data['First_Round'] = 0
    data['Conf_Semis'] = 0
    data['Conf_Finals'] = 0
    data['Finals_Made'] = 0
    data['Finals_Won'] = 0

    for year in range(start_year, end_year + 1):
        print(f"Year {year} playoff success calculating")
        for player in data['Player'].unique():
            recent_years = data[(data['Player'] == player) & (data['Year'].astype(int).between(year-5, year-1))]
            if not recent_years.empty:
                data.loc[(data['Player'] == player) & (data['Year'] == str(year)), 'First_Round'] = recent_years['Playoffs'].apply(lambda x: 1 if pd.notna(x) else 0).sum()
                data.loc[(data['Player'] == player) & (data['Year'] == str(year)), 'Conf_Semis'] = recent_years['Playoffs'].apply(lambda x: 1 if pd.notna(x) and ('Conf. Semis' in x or 'Conf. Finals' in x or 'Lost Finals' in x or 'Won Finals' in x) else 0).sum()
                data.loc[(data['Player'] == player) & (data['Year'] == str(year)), 'Conf_Finals'] = recent_years['Playoffs'].apply(lambda x: 1 if pd.notna(x) and ('Conf. Finals' in x or 'Lost Finals' in x or 'Won Finals' in x) else 0).sum()
                data.loc[(data['Player'] == player) & (data['Year'] == str(year)), 'Finals_Made'] = recent_years['Playoffs'].apply(lambda x: 1 if pd.notna(x) and ('Lost Finals' in x or 'Won Finals' in x) else 0).sum()
                data.loc[(data['Player'] == player) & (data['Year'] == str(year)), 'Finals_Won'] = recent_years['Playoffs'].apply(lambda x: 1 if pd.notna(x) and 'Won Finals' in x else 0).sum()

    return data

# Load the datasets
roster_data = load_data('roster_data_updated_f.csv')
salaries_data = load_data('salaries_data_final.csv')
team_history_data = load_data('team_history_data.csv')
shooting_data = load_data('nba_player_seasons_shooting_1980_2024.csv')
per_100_data = load_data('nba_player_seasons_100_1980_2024_rtg.csv')
adv_data = load_data('nba_player_seasons_advanced_1980_2024.csv')
per_game_data = load_data('nba_player_seasons_1978_2024.csv')
draft_data = load_data('nba_draft_data_1964_2024.csv')

# Extract and convert 'Year' column in the datasets
team_history_data = extract_and_convert_year(team_history_data, 'Season', 'Year')
salaries_data = extract_and_convert_year(salaries_data, 'season_end', 'Year')
shooting_data = convert_year_column(shooting_data, 'Year')
per_100_data = convert_year_column(per_100_data, 'Year')
adv_data = convert_year_column(adv_data, 'Year')
per_game_data = convert_year_column(per_game_data, 'Year')
draft_data = convert_year_column(draft_data, 'Year')

# Rename columns to ensure consistency for merging
salaries_data = rename_columns(salaries_data, {'name': 'Player'})
team_history_data = rename_columns(team_history_data, {'TeamAbR': 'Team'})
shooting_data = rename_columns(shooting_data, {'Tm': 'Team'})
per_100_data = rename_columns(per_100_data, {'Tm': 'Team'})
adv_data = rename_columns(adv_data, {'Tm': 'Team'})
per_game_data = rename_columns(per_game_data, {'Tm': 'Team'})
draft_data = rename_columns(draft_data, {'Tm': 'Team'})

# Ensure 'Team' column exists in datasets
for df in [salaries_data, shooting_data, per_100_data, adv_data, per_game_data, draft_data]:
    if 'Team' not in df.columns:
        df['Team'] = None

# Clean player names
roster_data = clean_player_names(roster_data)
salaries_data = clean_player_names(salaries_data)
shooting_data = clean_player_names(shooting_data)
per_100_data = clean_player_names(per_100_data)
adv_data = clean_player_names(adv_data)
per_game_data = clean_player_names(per_game_data)

# Remove rows where team is 'TOT'
roster_data = remove_tot_teams(roster_data)
salaries_data = remove_tot_teams(salaries_data)
shooting_data = remove_tot_teams(shooting_data)
per_100_data = remove_tot_teams(per_100_data)
adv_data = remove_tot_teams(adv_data)
per_game_data = remove_tot_teams(per_game_data)

# Reset the column names for merging purposes
roster_data.columns = roster_data.columns.str.replace('_roster', '')
salaries_data.columns = salaries_data.columns.str.replace('_salaries', '')
shooting_data.columns = shooting_data.columns.str.replace('_shooting', '')
per_100_data.columns = per_100_data.columns.str.replace('_per_100', '')
adv_data.columns = adv_data.columns.str.replace('_adv', '')
per_game_data.columns = per_game_data.columns.str.replace('_per_game', '')
draft_data.columns = draft_data.columns.str.replace('_draft', '')

# Convert 'Year' column in both DataFrames to string
roster_data['Year'] = roster_data['Year'].astype(str)
salaries_data['Year'] = salaries_data['Year'].astype(str)

# Merge roster_data with team_history_data first
team_history_data['Year'] = (team_history_data['Year'].astype(int) + 1).astype(str)
merged_data = pd.merge(roster_data, team_history_data, on=['Team', 'Year'], how='left')

# Calculate playoff success starting from 1985
merged_data = calculate_playoff_success(merged_data, 1985, merged_data['Year'].astype(int).max())

# Merge the rest of the datasets
merged_data = pd.merge(merged_data, salaries_data, on=['Player', 'Year'], how='left', suffixes=('', '_salaries'))
merged_data = pd.merge(merged_data, shooting_data, on=['Player', 'Team', 'Year'], how='left', suffixes=('', '_shooting'))
merged_data = pd.merge(merged_data, per_100_data, on=['Player', 'Team', 'Year'], how='left', suffixes=('', '_per_100'))
merged_data = pd.merge(merged_data, adv_data, on=['Player', 'Team', 'Year'], how='left', suffixes=('', '_adv'))
merged_data = pd.merge(merged_data, per_game_data, on=['Player', 'Team', 'Year'], how='left', suffixes=('', '_per_game'))

# Merge draft data separately on Player
try:
    merged_data = pd.merge(merged_data, draft_data, on=['Player'], how='left', suffixes=('', '_draft'))
except KeyError as e:
    print(f"Error during merging draft data: {e}")
    exit()

# Drop the last 17 columns
merged_data = merged_data.iloc[:, :-17]

# Transliterate special characters to their closest English equivalents before saving
for col in merged_data.select_dtypes(include=[object]).columns:
    merged_data[col] = merged_data[col].map(lambda x: unidecode(x) if isinstance(x, str) else x)

# Save the final merged data
output_file = 'combined_nba_data_fin.csv'
merged_data.to_csv(output_file, index=False)
print(f"Combined data has been saved to '{output_file}'")
