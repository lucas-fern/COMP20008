import matplotlib.pyplot as plt
from numpy import arange
from bs4 import BeautifulSoup
import pandas
import requests
import json
import re


def task_1(root_url='http://comp20008-jh.eng.unimelb.edu.au:9889/main/'):
    '''Crawls the web beginning at the root URL and extracts the HTML heading
    text, storing it alongside the URL in a CSV file, task1.csv.'''
    # Initialise sets to store URLs so set operations can remove duplicates.
    links = {root_url}
    visited_links = set()
    pages = pandas.DataFrame(columns=['url', 'headline'])

    # Looping over all unvisited URLs starting at the root.
    while links - visited_links:
        url = (links - visited_links).pop()

        # Request site data and raise an exception if unsuccessful.
        # Adapted from https://realpython.com/python-requests/#the-get-request
        try:
            req = requests.get(url)
            req.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
        except Exception as err:
            print(f'Other error occurred: {err}')

        # Extract the article title and any links to other pages.
        soup = BeautifulSoup(req.text, 'lxml')
        headline = soup('h1')[0].text
        new_links = [root_url + link.get('href') for link in soup('a')]

        # Store the URL and title in a DataFrame.
        if url != root_url:
            pages = pages.append({'url': url, 'headline': headline},
                                 ignore_index=True)

        links = links.union(set(new_links))
        visited_links.add(url)

    pages.to_csv('task1.csv', index=False)


def task_2(player_data_json='tennis.json', pages_csv='task1.csv'):
    '''Extracts the first complete player name and match score present in each
    of webpages crawled in task_1 and produces a CSV file containing the url,
    headline, player and score.'''
    # Extract all the player names from tennis.json.
    with open(player_data_json) as read_file:
        player_data = json.load(read_file)
    names = [player['name'].lower() for player in player_data]

    # Load the data from task_1.
    pages = pandas.read_csv(pages_csv)

    # Iterate over all URLs crawled.
    for url in pages.loc[:, 'url']:
        # Request site data and raise an exception if unsuccessful.
        # Adapted from https://realpython.com/python-requests/#the-get-request
        try:
            req = requests.get(url)
            req.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
        except Exception as err:
            print(f'Other error occurred: {err}')

        soup = BeautifulSoup(req.text, 'lxml')

        # Extract all the text from the webpage and search it for player names.
        text = ' '.join([element.get_text() for element in soup(['p', 'h1'])])
        text = text.lower()
        names_regex = '((' + ')|('.join(names) + '))'
        player_regex = re.compile(names_regex)
        player = re.search(player_regex, text)

        # Search the page for a valid game score according to valid_score().
        score_regex = re.compile(r'(((\d+-\d+)|(\(\d+[-/]\d+\)))[. ,]){2,}')
        score_iter = re.finditer(score_regex, text)

        page_score = None
        for game_score in score_iter:
            if valid_score(game_score):
                page_score = game_score
                break

        # Add players with valid scores to a DataFrame.
        if player and page_score:
            name = player.group().title()
            score = page_score.group()
            pages.loc[pages['url'] == url, 'player'] = name
            pages.loc[pages['url'] == url, 'score'] = \
                score.replace(',', ' ').replace('/', '-').strip(',.  ')
        else:
            pages = pages.loc[pages['url'] != url]

    pages.to_csv('task2.csv', index=False)


def valid_score(scores):
    '''Verifies that a regular expression object contains a complete and valid
    tennis game score where one player has won either 2 or 3 sets.'''
    # Remove any superfluous characters from scores.
    score_list = scores.group().strip(',.  ').replace(',', ' ').split()
    score_list = [x.split('-') for x in score_list if '(' and ')' not in x]

    # Count a win if a player scores >=6 and more than the other player.
    p1_wins = p2_wins = 0
    for score in score_list:
        score = [int(x) for x in score]
        if score[0] >= 6 and score[0] >= score[-1]:
            p1_wins += 1
        elif score[-1] >= 6 and score[-1] >= score[0]:
            p2_wins += 1
        else:
            return False

    # Only validate a game score if it was a best of 3 or 5.
    if (p1_wins in (2, 3) or p2_wins in (2, 3)) and p1_wins != p2_wins:
        return True

    return False


def task_3(pages_csv='task2.csv'):
    '''Identifies the game difference for each game from which a valid player
    and score was found in task_2, averages the game differences for each
    player, and saves a CSV with each player and their avg_game_difference.'''
    # Reads the score data from task_2.
    pages = pandas.read_csv(pages_csv)

    # Sum the game difference over all sets and add to a DataFrame.
    for game_score in pages.loc[:, 'score']:
        game_difference = 0
        score_list = [x.split('-') for x in game_score.split()
                      if '(' and ')' not in x]

        for set_score in score_list:
            set_score = [int(x) for x in set_score]
            game_difference += set_score[0] - set_score[-1]

        pages.loc[pages['score'] == game_score, 'game_difference'] = \
            abs(game_difference)

    # Average each players game differences if multiple recorded games.
    avg_game_dif = pages.loc[:, ['player', 'game_difference']].groupby(
        'player').mean()

    avg_game_dif.rename(
        columns={'game_difference': 'avg_game_difference'}).to_csv('task3.csv')


def task_4(pages_csv='task2.csv'):
    '''Generates a plot of the players who have the top 5 number of articles
    written about them.'''
    pages = pandas.read_csv(pages_csv)

    # Use mergesort to preserve alphabetical player order as it is stable.
    article_count = pages.groupby('player').count()\
        .sort_values(by=['url'], ascending=False, kind='mergesort')['url']

    top_5_articles = article_count.head(5)

    top_5_articles.plot(kind='bar')
    plt.title('Top 5 Players by Articles Written About Them')
    plt.xlabel('Player')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig('task4.png')


def task_5(player_data_json='tennis.json', pages_csv='task3.csv'):
    '''Generates a plot of all players identified in task_2 and their
    win percentage.'''
    avg_game_difs = pandas.read_csv(pages_csv, index_col='player')

    # Extract all the players win percentages from tennis.json.
    with open(player_data_json) as read_file:
        player_data = json.load(read_file)

    for player in avg_game_difs.index:
        for player_stats in player_data:
            if player_stats['name'].title() == player:
                avg_game_difs.loc[player, 'Win Percent'] =\
                    float(player_stats['wonPct'].strip('%'))

    # Plot, clean and label graph from DataFrame.
    avg_game_difs.rename(inplace=True,
                         columns={'avg_game_difference':
                                  'Average Game Difference'})
    avg_game_difs.sort_values(by='Win Percent', inplace=True, ascending=False)
    avg_game_difs.plot(kind='bar', secondary_y='Win Percent')

    axis_1, axis_2 = plt.gcf().get_axes()
    plt.title('''Average Game Difference and Win Percentage of
              all Players Cited in Articles''')
    axis_1.set_ylabel('Average Game Difference')
    axis_2.set_ylabel('Win Percentage')

    plt.tight_layout()
    plt.savefig('task5.png')


task_1()
task_2()
task_3()
task_4()
task_5()
