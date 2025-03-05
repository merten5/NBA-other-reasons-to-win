import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

### per game

def get_player_seasons(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    attempts = 0
    max_attempts = 5
    
    while attempts < max_attempts:
        response = requests.get(url)
        if response.status_code == 200:
            break
        elif response.status_code == 429:
            attempts += 1
            wait_time = 10 + 5 * attempts
            print(f"Rate limited. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        else:
            print(f"Failed to retrieve data for the year {year}. HTTP Status code: {response.status_code}")
            return [], []

    if response.status_code != 200:
        return [], []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table with player stats
    table = soup.find('table', {'id': 'per_game_stats'})
    
    if table is None:
        print(f"No table found for the year {year}")
        return [], []
    
    # Get all the rows from the table
    rows = table.find_all('tr')
    
    # Extract column headers
    headers = [th.getText() for th in rows[0].find_all('th')][1:]  # Skip the first header (Rk)
    
    # Extract player data
    player_data = []
    for row in rows[1:]:  # Skip the header row
        if row.find('th', {"scope": "row"}) is not None:  # Skip rows without player data
            player_stats = [td.getText() for td in row.find_all('td')]
            player_data.append(player_stats)
    
    return headers, player_data

def scrape_player_seasons(start_year=1980, end_year=2024, time_limit=900):
    start_time = time.time()
    all_player_data = []
    headers = None  # Initialize headers variable
    
    for year in range(start_year, end_year + 1):
        if time.time() - start_time > time_limit:
            print("Time limit exceeded. Stopping the scraping process.")
            break
        
        print(f"Scraping data for the {year} season...")
        year_headers, player_data = get_player_seasons(year)
        
        if not player_data:
            continue
        
        if headers is None and year_headers:
            headers = ['Year'] + year_headers
        
        for player in player_data:
            all_player_data.append([year] + player)
        
        # Pause to avoid overwhelming the server
        time.sleep(5)
    
    if headers is None:
        print("No headers found. No data was scraped.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    # Create a DataFrame
    df = pd.DataFrame(all_player_data, columns=headers)
    
    return df

# Scrape data from 1980 to the current year with a time limit of 300 seconds
df = scrape_player_seasons()

# Save to a CSV file with a unique name to avoid permission issues
output_file = 'nba_player_seasons_1980_2024.csv'
counter = 1
while os.path.exists(output_file):
    output_file = f'nba_player_seasons_1980_2024_{counter}.csv'
    counter += 1

df.to_csv(output_file, index=False)
print(f"Data has been scraped and saved to '{output_file}'")


### adv

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from unidecode import unidecode

def get_player_seasons_adv(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    attempts = 0
    max_attempts = 5

    while attempts < max_attempts:
        response = requests.get(url)
        if response.status_code == 200:
            break
        elif response.status_code == 429:
            attempts += 1
            wait_time = 10 + 5 * attempts
            print(f"Rate limited. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        else:
            print(f"Failed to retrieve data for the year {year}. HTTP Status code: {response.status_code}")
            return [], []

    if response.status_code != 200:
        return [], []

    soup = BeautifulSoup(response.content, 'html.parser')
    ##print(soup)

    # Find the table with player stats
    table = soup.find('table', {'id': 'advanced_stats'})
    
    if table is None:
        print(f"No table found for the year {year}")
        print(soup)  # Print soup for debuggign
        return [], []

    # Get all the rows from the table
    rows = table.find_all('tr')

    # Extract column headers
    headers = [th.getText() for th in rows[0].find_all('th')][1:]  # Skip the first header (Rk)

    # Extract player data
    player_data = []
    for row in rows[1:]:  # Skip the header row
        if row.find('th', {"scope": "row"}) is not None:  # Skip rows without player data
            player_stats = [unidecode(td.getText()) for td in row.find_all('td')]
            player_data.append(player_stats)

    return headers, player_data

def scrape_player_seasons_adv(start_year=2005, end_year=2024, time_limit=300):
    start_time = time.time()
    all_player_data = []
    headers = None  # Initialize headers variable

    for year in range(start_year, end_year + 1):
        if time.time() - start_time > time_limit:
            print("Time limit exceeded. Stopping the scraping process.")
            break

        print(f"Scraping data for the {year} season...")
        year_headers, player_data = get_player_seasons_adv(year)

        if not player_data:
            continue

        if headers is None and year_headers:
            headers = ['Year'] + year_headers

        for player in player_data:
            all_player_data.append([year] + player)

        # Pause to avoid overwhelming the server
        time.sleep(5)

    if headers is None:
        print("No headers found. No data was scraped.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Create a DataFrame
    df = pd.DataFrame(all_player_data, columns=headers)

    return df

# Scrape data from 1980 to the current year with a time limit of 300 seconds
df = scrape_player_seasons_adv()

# Save to a CSV file with a unique name to avoid permission issues
output_file = 'nba_player_seasons_advanced_2005_2024.csv'
counter = 1
while os.path.exists(output_file):
    output_file = f'nba_player_seasons_advanced_2005_2024_{counter}.csv'
    counter += 1

# Transliterate special characters before saving
for col in df.select_dtypes(include=[object]).columns:
    df[col] = df[col].map(lambda x: unidecode(x) if isinstance(x, str) else x)

df.to_csv(output_file, index=False)
print(f"Data has been scraped and saved to '{output_file}'")


### adj_shooting
import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
import os
from unidecode import unidecode

def get_player_seasons_shooting(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_adj_shooting.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the commented section that contains the table
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    table = None
    for comment in comments:
        if 'table_container' in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', {'id': 'adj-shooting'})
            if table:
                print("Table found!")
                break

    # If the table is found, you can proceed to parse it
    if table:
        headers = [th.get_text() for th in table.find('thead').find_all('th')]
        rows = []
        for tr in table.find('tbody').find_all('tr'):
            cells = [td.get_text() for td in tr.find_all('td')]
            rows.append(cells)

        # Display the headers and a few rows to verify
        print(headers)
        for row in rows[:5]:
            print(row)
    else:
        print(f"No table found for the year {year}")
        return [], []
    
    # Get all the rows from the table
    rows = table.find_all('tr')
    
    # Extract column headers, skipping the 'over_header' row
    headers = [th.getText() for th in rows[1].find_all('th')][1:]  # Skip the first header (Rk)
    
    # Extract player data
    player_data = []
    for row in rows[2:]:  # Skip the header rows
        if row.find('th', {"scope": "row"}) is not None:  # Skip rows without player data
            player_stats = [unidecode(td.getText()) for td in row.find_all('td')]
            player_data.append(player_stats)
    
    return headers, player_data

def scrape_player_seasons_shooting(start_year=1981, end_year=2024, time_limit=300):
    start_time = time.time()
    all_player_data = []
    headers = None  # Initialize headers variable
    
    for year in range(start_year, end_year + 1):
        if time.time() - start_time > time_limit:
            print("Time limit exceeded. Stopping the scraping process.")
            break
        
        print(f"Scraping data for the {year} season...")
        year_headers, player_data = get_player_seasons_shooting(year)
        
        if not player_data:
            continue
        
        if headers is None and year_headers:
            headers = ['Year'] + year_headers
        
        for player in player_data:
            all_player_data.append([year] + player)
        
        # Pause to avoid overwhelming the server
        time.sleep(5)
    
    if headers is None:
        print("No headers found. No data was scraped.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    # Create a DataFrame
    df = pd.DataFrame(all_player_data, columns=headers)
    
    return df

# Scrape data from 1980 to the current year with a time limit of 300 seconds
df = scrape_player_seasons_shooting()

# Save to a CSV file with a unique name to avoid permission issues
output_file = 'nba_player_seasons_shooting_1981_2024.csv'
counter = 1
while os.path.exists(output_file):
    output_file = f'nba_player_seasons_shooting_1981_2024_{counter}.csv'
    counter += 1

# Transliterate special characters before saving
for col in df.select_dtypes(include=[object]).columns:
    df[col] = df[col].map(lambda x: unidecode(x) if isinstance(x, str) else x)

df.to_csv(output_file, index=False)
print(f"Data has been scraped and saved to '{output_file}'")


### 100

import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import os

def get_player_seasons_100(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_poss.html"
    attempts = 0
    max_attempts = 5
    
    while attempts < max_attempts:
        response = requests.get(url)
        if response.status_code == 200:
            break
        elif response.status_code == 429:
            attempts += 1
            wait_time = 10 + 5 * attempts
            print(f"Rate limited. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        else:
            print(f"Failed to retrieve data for the year {year}. HTTP Status code: {response.status_code}")
            return [], []

    if response.status_code != 200:
        return [], []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table with player stats
    table = soup.find('table', {'id': 'per_poss_stats'})
    
    if table is None:
        print(f"No table found for the year {year}")
        return [], []
    
    # Get all the rows from the table
    rows = table.find_all('tr')
    
    # Extract column headers
    headers = [th.getText() for th in rows[0].find_all('th')][1:]  # Skip the first header (Rk)
    
    # Extract player data
    player_data = []
    for row in rows[1:]:  # Skip the header row
        if row.find('th', {"scope": "row"}) is not None:  # Skip rows without player data
            player_stats = [td.getText() for td in row.find_all('td')]
            player_data.append(player_stats)
    
    return headers, player_data

def scrape_player_seasons_100(start_year=1981, end_year=2024, time_limit=900):
    start_time = time.time()
    all_player_data = []
    headers = None  # Initialize headers variable
    
    for year in range(start_year, end_year + 1):
        if time.time() - start_time > time_limit:
            print("Time limit exceeded. Stopping the scraping process.")
            break
        
        print(f"Scraping data for the {year} season...")
        year_headers, player_data = get_player_seasons_100(year)
        
        if not player_data:
            continue
        
        if headers is None and year_headers:
            headers = ['Year'] + year_headers
        
        for player in player_data:
            all_player_data.append([year] + player)
        
        # Pause to avoid overwhelming the server
        time.sleep(5)
    
    if headers is None:
        print("No headers found. No data was scraped.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    # Create a DataFrame
    df = pd.DataFrame(all_player_data, columns=headers)
    
    # Select only relevant columns
    columns_to_keep = ['Year', 'Player', 'Pos', 'Age', 'Tm', 'ORtg', 'DRtg']
    df = df[columns_to_keep]
    
    return df

# Scrape data from 1980 to the current year with a time limit of 300 seconds
df = scrape_player_seasons_100()

# Save to a CSV file with a unique name to avoid permission issues
output_file = 'nba_player_seasons_100_1981_2024.csv'
counter = 1
while os.path.exists(output_file):
    output_file = f'nba_player_seasons_100_1981_2024_{counter}.csv'
    counter += 1

df.to_csv(output_file, index=False)
print(f"Data has been scraped and saved to '{output_file}'")


##### Team rosters and results?
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_team_data(url_template, team_abbreviation, year):
    url = url_template.format(team_abbreviation, year)
    attempts = 0
    max_attempts = 5

    while attempts < max_attempts:
        response = requests.get(url)
        if response.status_code == 200:
            break
        elif response.status_code == 429:
            attempts += 1
            wait_time = 10 + 5 * attempts
            print(f"Rate limited. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        else:
            print(f"Failed to retrieve data for the team {team_abbreviation} in year {year}. HTTP Status code: {response.status_code}")
            return [], [], [], []

    if response.status_code != 200:
        return [], [], [], []

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract team stats table
    team_stats_table = soup.find('table', {'id': 'team_and_opponent'})
    team_stats_headers, team_stats_data = [], []
    if team_stats_table:
        rows = team_stats_table.find_all('tr')
        team_stats_headers = [th.getText() for th in rows[0].find_all('th')][1:]  # Skip the first header (Rk)
        team_stats_data = [[td.getText() for td in row.find_all('td')] for row in rows[1:]]
    
    # Extract roster table
    roster_table = soup.find('table', {'id': 'roster'})
    roster_headers, roster_data = [], []
    if roster_table:
        rows = roster_table.find_all('tr')
        roster_headers = [th.getText() for th in rows[0].find_all('th')][1:]  # Skip the first header (Rk)
        roster_data = [[td.getText() for td in row.find_all('td')] for row in rows[1:]]

    return team_stats_headers, team_stats_data, roster_headers, roster_data

def scrape_teams(start_year=1964, end_year=1979, time_limit=300):
    start_time = time.time()
    all_team_stats = []
    all_roster_data = []
    team_stats_headers = None
    roster_headers = None
    
    url_template = "https://www.basketball-reference.com/teams/{}/{}.html"
    abbreviations = ["LAL", "GSW", "PHO", "BOS", "CHI", "WSB", "SEA", "IND", "SDC", "HOU", "NJN", "SAS", "POR", "MIL", "DET", "PHI", "KCK", "UTA", "DEN", "ATL", "CLE", "NYK", "DAL", "LAC", "SAC", "CHH", "MIA", "ORL", "MIN", "VAN", "TOR", "WAS", "MEM", "NOH", "CHA", "NOK", "OKC", "BRK", "NOP", "CHO"]

    for year in range(start_year, end_year + 1):
        if time.time() - start_time > time_limit:
            print("Time limit exceeded. Stopping the scraping process.")
            break
        
        print(f"Scraping data for the {year} season...")
        
        for team_abbreviation in abbreviations:
            print(f"Scraping data for team {team_abbreviation} in year {year}...")
            try:
                team_stats_headers_tmp, team_stats_data, roster_headers_tmp, roster_data = get_team_data(url_template, team_abbreviation, year)
            except ValueError:
                print(f"Skipping team {team_abbreviation} for year {year} due to repeated rate limiting or other issues.")
                continue
            
            if team_stats_data:
                print(f"Team stats data for {team_abbreviation} in {year}: {team_stats_data[0]}")  # Print first row for debugging
                if not team_stats_headers:
                    team_stats_headers = team_stats_headers_tmp
                for stats in team_stats_data:
                    all_team_stats.append([year, team_abbreviation] + stats)
            
            if roster_data:
                print(f"Roster data for {team_abbreviation} in {year}: {roster_data[0]}")  # Print first row for debugging
                if not roster_headers:
                    roster_headers = roster_headers_tmp
                for roster in roster_data:
                    all_roster_data.append([year, team_abbreviation] + roster)
        
        # Pause to avoid overwhelming the server
        time.sleep(5)
    
    if not all_team_stats:
        print("No team stats data was scraped.")
        team_stats_df = pd.DataFrame()
    else:
        team_stats_df = pd.DataFrame(all_team_stats, columns=["Year", "Team"] + team_stats_headers)
    
    if not all_roster_data:
        print("No roster data was scraped.")
        roster_df = pd.DataFrame()
    else:
        roster_df = pd.DataFrame(all_roster_data, columns=["Year", "Team"] + roster_headers)
    
    return team_stats_df, roster_df

# Example usage:
team_stats_df, roster_df = scrape_teams(start_year=1964, end_year=1979)
print("Team Stats DataFrame:")
print(team_stats_df.head())
print("Roster DataFrame:")
print(roster_df.head())

# Export the roster data to a CSV file
roster_df.to_csv("roster_data_pre.csv", index=False)
print("Roster data exported to 'roster_data_1964_1979.csv'.")

### Team history
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_team_history_data(team_abbr):
    url = f"https://www.basketball-reference.com/teams/{team_abbr}/#{team_abbr}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve data for {team_abbr}. HTTP Status code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table containing the history data
    history_table = soup.find('table', {'id': team_abbr})
    if not history_table:
        print(f"No history table found for {team_abbr}.")
        return []
    
    rows = history_table.find_all('tr')
    headers = [th.getText() for th in rows[0].find_all('th')]
    
    data = []
    for row in rows[1:]:
        cols = row.find_all(['th', 'td'])  # Include 'th' for the season data
        row_data = [col.getText().strip() for col in cols]
        if row_data:  # Only add rows that have data
            data.append([team_abbr] + row_data)
    
    return headers, data

def scrape_all_teams_history_data(team_abbrs, time_limit=300):
    start_time = time.time()
    all_data = []
    max_columns = 0  # Track the maximum number of columns across all teams
    headers = None
    
    for team_abbr in team_abbrs:
        if time.time() - start_time > time_limit:
            print("Time limit exceeded. Stopping the scraping process.")
            break
        
        print(f"Scraping history data for {team_abbr}...")
        
        try:
            headers_tmp, team_data = get_team_history_data(team_abbr)
            if headers_tmp and not headers:
                headers = headers_tmp
            if headers_tmp and len(headers_tmp) > max_columns:
                max_columns = len(headers_tmp)
            for game in team_data:
                all_data.append(game)
        except Exception as e:
            print(f"Error scraping data for {team_abbr}: {e}")
            continue
        
        # Pause to avoid overwhelming the server
        time.sleep(5)
    
    if not all_data:
        print("No history data was scraped.")
        history_df = pd.DataFrame()
    else:
        # Ensure all rows have the same number of columns by padding with None
        standardized_data = []
        for row in all_data:
            if len(row) < max_columns + 1:  # +1 for the 'Team' column
                row.extend([None] * (max_columns + 1 - len(row)))
            standardized_data.append(row)
        
        headers = ["Team"] + headers
        history_df = pd.DataFrame(standardized_data, columns=headers)
    
    return history_df

# List of team abbreviations
team_abbrs = ['ATL', 	'BOS', 	'BRK', 	'CHA', 	'CHH', 	'CHI', 	'CHO', 	'CLE', 	'DAL', 	'DEN', 	'DET', 	'GSW', 	'HOU', 	'IND',
              'KCK', 	'LAC', 	'LAL', 	'MEM', 	'MIA', 	'MIL', 	'MIN', 	'NJN', 	'NOH', 	'NOK', 	'NOP', 	'NYK', 	'OKC', 	'ORL',
              'PHI', 	'PHO', 	'POR', 	'SAC', 	'SAS', 	'SDC', 	'SEA', 	'TOR', 	'UTA', 	'VAN', 	'WAS', 	'WSB']


# Example usage:
history_data_df = scrape_all_teams_history_data(team_abbrs)
print("history Data DataFrame:")
print(history_data_df.head())

# Export the history data to a CSV file
history_data_df.to_csv("team_history_data.csv", index=False)
print("history data exported to 'team_history_data.csv'.")

### Team Salaries filling in the blanks from the other data set.
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_salary_data(url, year='2024-2025'):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data from {url}. HTTP Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    data = []

    # Extract data for 2024-2025
    if '2024-2025' in url:
        salary_table = soup.find('table', {'class': 'hh-salaries-ranking-table'})
        if salary_table:
            rows = salary_table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cols = row.find_all('td')
                name = cols[1].get_text(strip=True)
                salary = cols[2].get('data-value', '0')
                data.append(['2024-2025', name, salary])
    else:
        # Extract data for other years
        salary_table = soup.find('table', {'class': 'hh-salaries-ranking-table'})
        if salary_table:
            rows = salary_table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cols = row.find_all('td')
                name = cols[1].get_text(strip=True)
                salary = cols[2].get('data-value', '0')
                data.append([f"{year}-{year+1}", name, salary])

    return data

def scrape_all_years(start_year, end_year):
    base_url = "https://hoopshype.com/salaries/players/"
    all_data = []

    for year in range(start_year, end_year + 1):
        url = f"{base_url}{year}-{year+1}/"
        print(f"Scraping data from {url}...")
        year_data = get_salary_data(url, year)
        all_data.extend(year_data)
    
    # For 2024-2025
    url = f"{base_url}2024-2025/"
    print(f"Scraping data from {url}...")
    year_data = get_salary_data(url)
    all_data.extend(year_data)

    return all_data

# Example usage
start_year = 2016
end_year = 2024
salary_data = scrape_all_years(start_year, end_year)

# Convert to DataFrame and export
df = pd.DataFrame(salary_data, columns=['Year', 'Name', 'Salary'])
df['Salary'] = df['Salary'].replace({'\$': '', ',': ''}, regex=True).astype(float)  # Clean and convert salary to float
df.to_csv('salaries_data.csv', index=False)
print("Salary data exported to 'salaries_data.csv'.")

import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_salary_data_2024_2025(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data from {url}. HTTP Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    print(soup)
    data = []

    # Extract data for 2024-2025
    salary_table = soup.find('table', {'class': 'hh-salaries-ranking-table'})
    if salary_table:
        rows = salary_table.find_all('tr')[1:]  # Skip header row
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 3:  # Ensure there are enough columns
                name = cols[1].get_text(strip=True)
                salary = cols[2].get('data-value', '0')
                data.append(['2024-2025', name, salary])

    return data

# URL for 2024-2025 salary data
url_2024_2025 = "https://hoopshype.com/salaries/players/"

# Scrape the data
salary_data = get_salary_data_2024_2025(url_2024_2025)

# Convert to DataFrame and export
df = pd.DataFrame(salary_data, columns=['Year', 'Name', 'Salary'])
df['Salary'] = df['Salary'].replace({'\$': '', ',': ''}, regex=True).astype(float)  # Clean and convert salary to float
df.to_csv('salaries_2024_2025.csv', index=False)
print("Salary data for 2024-2025 exported to 'salaries_2024_2025.csv'.")


###Updating roster.csv for rate limited skipped teams
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from unidecode import unidecode

def get_team_data(url_template, team_abbreviation, year, time_limit):
    url = url_template.format(team_abbreviation, year)
    start_time = time.time()
    attempts = 0
    max_attempts = 3600

    while True:
        response = requests.get(url)
        if response.status_code == 200:
            # Successfully retrieved data
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract team stats table
            team_stats_table = soup.find('table', {'id': 'team_and_opponent'})
            team_stats_headers, team_stats_data = [], []
            if team_stats_table:
                rows = team_stats_table.find_all('tr')
                team_stats_headers = [th.getText() for th in rows[0].find_all('th')][1:]  # Skip the first header (Rk)
                team_stats_data = [[td.getText() for td in row.find_all('td')] for row in rows[1:]]
            
            # Extract roster table
            roster_table = soup.find('table', {'id': 'roster'})
            roster_headers, roster_data = [], []
            if roster_table:
                rows = roster_table.find_all('tr')
                roster_headers = [th.getText() for th in rows[0].find_all('th')][1:]  # Skip the first header (Rk)
                roster_data = [[td.getText() for td in row.find_all('td')] for row in rows[1:]]

            return team_stats_headers, team_stats_data, roster_headers, roster_data, response.status_code
        
        elif response.status_code == 429:
            # Rate limited
            print(f"Rate limited. Attempt number {attempts}. Waiting for 121 seconds before retrying... HTTP Status code: {response.status_code}")
            time.sleep(121)
            if time.time() - start_time > time_limit:
                print("Time limit exceeded. Exiting.")
                return [], [], [], [], response.status_code
        
        elif response.status_code == 404:
            # Missing
            print(f"Failed to retrieve data for the team {team_abbreviation} in year {year}. HTTP Status code: {response.status_code}")
            print(f"Skipping this year due to 404 error.")
            return [], [], [], [], response.status_code
        
        else:
            # Other errors
            print(f"Failed to retrieve data for the team {team_abbreviation} in year {year}. HTTP Status code: {response.status_code}")
            attempts += 1
            wait_time = 10 + 5 * attempts
            if attempts >= max_attempts:
                print(f"Max attempts reached for {team_abbreviation} in year {year}. Skipping to the next year.")
                return [], [], [], [], response.status_code
            print(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)

def scrape_teams(start_year=1980, end_year=2024, time_limit=5400, existing_data=None):
    start_time = time.time()
    all_team_stats = []
    all_roster_data = []
    team_stats_headers = None
    roster_headers = None
    
    url_template = "https://www.basketball-reference.com/teams/{}/{}.html"
    abbreviations = ["BRK", "CHO", "DET", "KCK", "LAC", "MEM", "NYK", 
                     "SAC", "ORL", "MIN", "NOH", "CHA", "NOK", "BOS", "CHI", "WSB", "SEA", "IND",
                     "OKC", "NOP", "CHH", "MIA", "VAN", "TOR", "WAS", "HOU", "CLE", "GSW", "PHO",
                     "NJN", "ATL", "DAL", "LAL", "SDC", "SAS", "POR", "MIL", "PHI", "UTA", "DEN"]
                     
    
    existing_combinations = set()
    if existing_data is not None:
        existing_combinations = set(zip(existing_data['Year'], existing_data['Team']))

    def save_progress(all_roster_data, roster_headers):
        # Convert all_roster_data to DataFrame
        if not all_roster_data:
            roster_df = pd.DataFrame()
        else:
            roster_df = pd.DataFrame(all_roster_data, columns=["Year", "Team"] + roster_headers)
        
        # Transliterate special characters before saving
        for col in roster_df.select_dtypes(include=[object]).columns:
            roster_df[col] = roster_df[col].map(lambda x: unidecode(x) if isinstance(x, str) else x)

        # Combine the new roster data with the existing data
        combined_roster_df = pd.concat([existing_roster_data, roster_df]).drop_duplicates(subset=["Year", "Team", "Player"])

        # Export the combined roster data to a CSV file
        combined_roster_df.to_csv("roster_data_updated_f.csv", index=False)
        print("Combined roster data exported to 'roster_data_updated_f.csv'.")

    for team_abbreviation in abbreviations:
        if time.time() - start_time > time_limit:
            print("Total time limit exceeded. Stopping the scraping process.")
            save_progress(all_roster_data, roster_headers)
            break

        for year in range(start_year, end_year + 1):
            if (year, team_abbreviation) in existing_combinations:
                print(f"Skipping existing data for team {team_abbreviation} in year {year}...")
                continue
            
            print(f"Scraping data for team {team_abbreviation} in year {year}...")
            team_stats_headers_tmp, team_stats_data, roster_headers_tmp, roster_data, status_code = get_team_data(url_template, team_abbreviation, year, time_limit)

            # If rate limit error occurs, exit the process
            if status_code == 429:
                print("Stopping due to rate limit.")
                save_progress(all_roster_data, roster_headers)
                return pd.DataFrame(), pd.DataFrame()
            
            if team_stats_data:
                print(f"Team stats data for {team_abbreviation} in {year}: {team_stats_data[0]}")  # Print first row for debugging
                if not team_stats_headers:
                    team_stats_headers = team_stats_headers_tmp
                for stats in team_stats_data:
                    all_team_stats.append([year, team_abbreviation] + stats)
            
            if roster_data:
                print(f"Roster data for {team_abbreviation} in {year}: {roster_data[0]}")  # Print first row for debugging
                if not roster_headers:
                    roster_headers = roster_headers_tmp
                for roster in roster_data:
                    all_roster_data.append([year, team_abbreviation] + roster)

        # Pause to avoid overwhelming the server
        time.sleep(5)
    
    if not all_team_stats:
        print("No team stats data was scraped.")
        team_stats_df = pd.DataFrame()
    else:
        team_stats_df = pd.DataFrame(all_team_stats, columns=["Year", "Team"] + team_stats_headers)
    
    if not all_roster_data:
        print("No roster data was scraped.")
        roster_df = pd.DataFrame()
    else:
        roster_df = pd.DataFrame(all_roster_data, columns=["Year", "Team"] + roster_headers)

    save_progress(all_roster_data, roster_headers)
    
    return team_stats_df, roster_df

# Load the existing roster data
existing_roster_data = pd.read_csv('roster_data.csv', encoding='ISO-8859-1')

# Transliterate special characters to their closest English equivalents
for col in existing_roster_data.select_dtypes(include=[object]).columns:
    existing_roster_data[col] = existing_roster_data[col].map(lambda x: unidecode(x) if isinstance(x, str) else x)

# Example usage:
team_stats_df, roster_df = scrape_teams(start_year=1980, end_year=2024, time_limit=11000, existing_data=existing_roster_data)

#Draft info
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from unidecode import unidecode

def get_draft_data(year):
    url = f"https://www.basketball-reference.com/draft/NBA_{year}.html"
    attempts = 0
    max_attempts = 5

    while attempts < max_attempts:
        response = requests.get(url)
        if response.status_code == 200:
            break
        elif response.status_code == 429:
            attempts += 1
            wait_time = 10 + 5 * attempts
            print(f"Rate limited. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        else:
            print(f"Failed to retrieve data for the year {year}. HTTP Status code: {response.status_code}")
            return [], []

    if response.status_code != 200:
        return [], []

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table with draft stats
    table = soup.find('table', {'id': 'stats'})

    if table is None:
        print(f"No table found for the year {year}")
        return [], []

    # Get all the rows from the table
    rows = table.find_all('tr')

    # Extract column headers
    headers = [th.getText() for th in rows[0].find_all('th')]

    # Extract player data
    player_data = []
    for row in rows[1:]:  # Skip the header row
        if row.find('th', {"scope": "row"}) is not None:  # Skip rows without player data
            player_stats = [unidecode(td.getText()) for td in row.find_all('td')]
            player_data.append(player_stats)

    return headers, player_data

def scrape_draft_data(start_year=2024, end_year=2025, save_interval=2*60*60):
    start_time = time.time()
    all_player_data = []
    headers = ["Rk", "Pk", "Tm", "Player", "College", "Yrs", "Games", "Minutes Played", 
    "Points", "Total Rebounds", "Assists", "Field Goal Percentage", "3-Point Field Goal Percentage",
    "Free Throw Percentage", "Minutes Played Per Game", "Points Per Game", 
    "Total Rebounds Per Game", "Assists Per Game", "Win Shares", 
    "Win Shares Per 48 Minutes", "BPM", "VORP"]
    output_file = 'nba_draft_data_1999_2025.csv'
    counter = 1
    
    for year in range(start_year, end_year + 1):
        current_time = time.time()
        if current_time - start_time > save_interval:
            # Save interim data
            interim_df = pd.DataFrame(all_player_data, columns=headers if headers else [])
            while os.path.exists(output_file):
                output_file = f'nba_draft_data_1964_2024_{counter}.csv'
                counter += 1
            # Transliterate special characters before saving
            for col in interim_df.select_dtypes(include=[object]).columns:
                interim_df[col] = interim_df[col].map(lambda x: unidecode(x) if isinstance(x, str) else x)
            interim_df.to_csv(output_file, index=False)
            print(f"Interim data saved to '{output_file}' after {save_interval // 3600} hours.")
            break

        print(f"Scraping draft data for the {year} draft...")
        year_headers, player_data = get_draft_data(year)

        if not player_data:
            continue

        if headers is None and year_headers:
            headers = ['Year'] + year_headers
        elif headers and year_headers and headers[1:] != year_headers:
            print(f"Header mismatch for year {year}, adjusting headers")
            current_headers = ['Year'] + year_headers
            if len(current_headers) > len(headers):
                headers += [f"extra_{i}" for i in range(len(headers), len(current_headers))]
            elif len(headers) > len(current_headers):
                current_headers += [f"extra_{i}" for i in range(len(current_headers), len(headers))]

        for player in player_data:
            all_player_data.append([year] + player + [""] * (len(headers) - len(player) - 1))

        # Pause to avoid overwhelming the server
        time.sleep(1)

    if headers is None:
        print("No headers found. No data was scraped.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Create a DataFrame with all the data
    df = pd.DataFrame(all_player_data, columns=headers)

    return df

# Scrape draft data from 1964 to 2024
df = scrape_draft_data()

# Save to a CSV file with a unique name to avoid permission issues
output_file = 'nba_draft_data_1999_2025.csv'
counter = 1
while os.path.exists(output_file):
    output_file = f'nba_draft_data_1999_2025{counter}.csv'
    counter += 1

# Transliterate special characters before saving
for col in df.select_dtypes(include=[object]).columns:
    df[col] = df[col].map(lambda x: unidecode(x) if isinstance(x, str) else x)

df.to_csv(output_file, index=False)
print(f"Data has been scraped and saved to '{output_file}'")

