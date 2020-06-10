import pandas
import textdistance
import random


def blockless_link(amazon='amazon_small.csv', google='google_small.csv'):
    """Attemps record linkage without blocking of Amazon and Google
    product records.
    """

    # Weigts that achieved the best precision and recall when tested against
    # a training dataset
    title_weight = 3.9
    manuf_weight = 0.05
    price_weight = 0.35
    match_threshold = 1

    amazon_products = pandas.read_csv(amazon)
    google_products = pandas.read_csv(google)

    matches = pandas.DataFrame(
        columns=['idAmazon', 'idGoogleBase', 'score'])
    matches.set_index('idAmazon', inplace=True)

    # Iterate over every combination of Amazon and Google products
    for _, a_row in amazon_products.iterrows():
        a_title = a_row['title']
        a_id = a_row['idAmazon']

        for _, g_row in google_products.iterrows():
            g_title = g_row['name']
            g_id = g_row['idGoogleBase']

            # The score of any given match begins at zero...
            match_score = 0

            # ...is increased depending on the similarity of the titles...
            # (we raise this score to ^1.5 to favour the closest matches)
            if (type(a_title) == str and type(g_title) == str):
                match_score += (1 - textdistance.overlap.normalized_distance(
                    a_title.split(), g_title.split()))**1.5 * title_weight

            # ...is increased if the manufacturer appears similar...
            if (type(a_row['manufacturer']) == str
                    and type(g_row['manufacturer']) == str):
                match_score += textdistance.jaro_winkler(
                    a_row['manufacturer'], g_row['manufacturer']
                ) * manuf_weight

            # ...and decreases if the prices are not similar.
            if (a_row['price'] > 0 and g_row['price'] > 0):
                match_score -= abs(
                    (a_row['price']-g_row['price'])/a_row['price']
                ) * price_weight

            # If the score of the match is above the determined threshold for
            # a likely match, record it. If a match exists for that record
            # only record if the score is greater.
            if match_score >= match_threshold:
                if a_id not in list(matches.index):
                    matches = matches.append(
                        pandas.Series({'idGoogleBase': g_id,
                                       'score': match_score}, name=a_id))
                elif (matches.loc[a_id]['score'] < match_score):
                    matches['idGoogleBase'][a_id] = g_id
                    matches['score'][a_id] = match_score

    matches.drop('score', axis=1, inplace=True)
    matches.to_csv('task1a.csv')


blockless_link(amazon=r'ass2\task1\amazon_small.csv',
               google=r'ass2\task1\google_small.csv')
