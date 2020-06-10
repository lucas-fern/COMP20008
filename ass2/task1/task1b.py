import nltk
import pandas
import random
import string
import textdistance
from nltk.corpus import stopwords
nltk.download('stopwords')


def blockify(blocks, amazon='amazon.csv', google='google.csv'):
    """Attemps record linkage with blocking of Amazon and Google
    product records.
    """

    amazon_products = pandas.read_csv(amazon, index_col='idAmazon')
    google_products = pandas.read_csv(google, index_col='id')

    amazon_blocks = pandas.DataFrame(columns=['block_key', 'product_id'])
    google_blocks = pandas.DataFrame(columns=['block_key', 'product_id'])
    amazon_blocks = amazon_blocks.set_index('block_key')
    google_blocks = google_blocks.set_index('block_key')

    google_title_text = ' '.join(list(google_products['name']))
    for block in blocks:
        if google_title_text.count(block) < 200:
            for g_id, g_product in google_products.iterrows():
                if block in g_product['name']:
                    google_blocks = google_blocks.append(
                        pandas.Series({'product_id': g_id},
                                    name=block)
                    )

            for a_id, a_product in amazon_products.iterrows():
                if block in a_product['title']:
                    amazon_blocks = amazon_blocks.append(
                        pandas.Series({'product_id': a_id},
                                    name=block)
                    )

    amazon_blocks.to_csv('amazon_blocks.csv')
    google_blocks.to_csv('google_blocks.csv')


def generate_blocks(google='google.csv'):
    google_products = pandas.read_csv(google, index_col='id')
    words = ' '.join(list(google_products['name'])).split()
    words = [word for word in words if word.isalpha()]
    words_no_sw = [word for word in words if word not in stopwords.words()]
    blocks = [word for word in words_no_sw if words_no_sw.count(word) > 10]

    return list(set(blocks))


blocks = ['landscape', 'advanced', 'elementary', 'year', 'nintendo', 'image', 'deluxe', 'years', 'bundle', 'secure', 'laptops', 'master', 'success', 'hp', 'kit', 'architect', 'jumpstart', 'desktop', 'corporation', 'quicken', 'live', 'book', 'pc', 'cs', 'watchguard', 'video', 'protection', 'adventure', 'premier', 'email', 'spanish', 'easy', 'servers', 'pack', 'vpn', 'pro', 'suite', 'norton', 'authoring', 'school', 'individual', 'corel', 'professor', 'card', 'typing', 'insignia', 'llc', 'preschool', 'xbox', 'quick', 'autodesk', 'ipod', 'services', 'pianosoft', 'photo', 'abacus', 'discount', 'aspyr', 'development', 'effects', 'zone', 'laplink', 'cisco', 'yamaha', 'math', 'software', 'sweeper', 'retrospect', 'system', 'vista', 'total', 'creative', 'acrobat', 'security', 'play', 'games', 'time', 'office', 'educational', 'window', 'envelopes', 'additional', 'quickbooks', 'cd', 'dvd', 'support', 'kids', 'hoyle', 'higher', 'network', 'game', 'tax', 'tlp', 'securid', 'john', 'ios', 'iii', 'shop', 'global', 'inc', 'enterprise', 'gold', 'unlimited', 'sage', 'encore', 'collection', 'technology', 'magix', 'full', 'ext', 'photoshop', 'animation', 'sound', 'adobe', 'expansion', 'sb', 'flash', 'set', 'small', 'greatest', 'playstation', 'brain', 'filemaker', 'screen', 'technical', 'workshop', 'smart', 'guitar', 'power', 'performa', 'art', 'retail', 'windows', 'server', 'word', 'free', 'microsoft', 'package', 'emedia', 'puzzle', 'box', 'mgr', 'professional', 'express', 'truprevent', 'mnt',
          'instant', 'guide', 'premium', 'freeverse', 'upgrade', 'home', 'reader', 'users', 'single', 'spy', 'mobile', 'premiere', 'logic', 'auth', 'storage', 'tree', 'immersion', 'hits', 'management', 'open', 'platinum', 'series', 'production', 'project', 'publishing', 'digital', 'panda', 'appl', 'option', 'aquarium', 'steinberg', 'symantec', 'music', 'media', 'marketing', 'internet', 'world', 'systems', 'engineering', 'exec', 'architectural', 'english', 'tools', 'x', 'remote', 'gatdef', 'topics', 'fogware', 'visual', 'star', 'xp', 'mike', 'audio', 'cut', 'homespun', 'creator', 'forms', 'learn', 'simply', 'websphere', 'mcafee', 'print', 'ultra', 'antivirus', 'business', 'avanquest', 'interactive', 'sonicwall', 'final', 'sos', 'plus', 'finance', 'nt', 'learning', 'upg', 'basic', 'super', 'webroot', 'nova', 'win', 'standard', 'family', 'license', 'client', 'aggregation', 'version', 'big', 'data', 'subscription', 'macintosh', 'vmware', 'ibm', 'best', 'manager', 'user', 'allume', 'onone', 'accounting', 'corp', 'backup', 'ae', 'partners', 'emc', 'web', 'design', 'laser', 'intuit', 'match', 'std', 'continuous', 'ultimate', 'editing', 'piano', 'entertainment', 'agent', 'edition', 'adventures', 'grade', 'punch', 'compatible', 'apple', 'linux', 'elements', 'product', 'svr', 'studio', 'usa', 'company', 'volume', 'teaches', 'wasp', 'limited', 'academic', 'maker', 'tycoon', 'essentials', 'leonard', 'movie', 'mac', 'maintenance', 'checks', 'advantage', 'sony', 'sibelius', 'complete', 'csdc', 'service']
# blocks_2 = [b for b in blocks if blocks.count(b) < 100]
blocks_2 = ['greatest', 'vpn', 'success', 'vista', 'freeverse', 'global', 'advantage', 'watchguard', 'word', 'company', 'cut', 'engineering', 'data', 'series', 'steinberg', 'landscape', 'live', 'total', 'servers', 'sweeper', 'puzzle', 'ipod', 'macintosh', 'manager', 'platinum', 'game', 'visual', 'service', 'typing', 'volume', 'aquarium', 'discount', 'project', 'basic', 'best', 'spy', 'sibelius', 'laptops', 'flash', 'sos', 'quicken', 'ae', 'express', 'hits', 'apple', 'architect', 'authoring', 'std', 'editing', 'csdc', 'server', 'book', 'storage', 'playstation', 'simply', 'publishing', 'world', 'learn', 'collection', 'webroot', 'accounting', 'star', 'mnt', 'sonicwall', 'enterprise', 'exec', 'wasp', 'mike', 'sony', 'match', 'system', 'learning', 'acrobat', 'immersion', 'full', 'unlimited', 'retail', 'set', 'premium', 'smart', 'performa', 'tax', 'super', 'photoshop', 'individual', 'agent', 'higher', 'internet', 'limited', 'tree', 'instant', 'entertainment', 'gold', 'games', 'family', 'creative', 'linux', 'guitar', 'web', 'pianosoft', 'professor', 'elementary', 'security', 'final', 'school', 'securid', 'punch', 'preschool', 'free', 'desktop', 'elements', 'leonard', 'norton', 'emc', 'animation', 'photo', 'power', 'vmware', 'tlp', 'time', 'corporation', 'gatdef', 'panda', 'marketing', 'onone', 'single', 'llc', 'client',
            'support', 'hp', 'sb', 'continuous', 'finance', 'additional', 'architectural', 'kit', 'abacus', 'secure', 'spanish', 'management', 'aggregation', 'maintenance', 'ultimate', 'studio', 'fogware', 'adventures', 'retrospect', 'truprevent', 'topics', 'movie', 'audio', 'years', 'sage', 'expansion', 'usa', 'ultra', 'interactive', 'digital', 'remote', 'cisco', 'tycoon', 'effects', 'aspyr', 'insignia', 'technology', 'guide', 'math', 'corel', 'small', 'mgr', 'xbox', 'quick', 'logic', 'version', 'network', 'video', 'advanced', 'image', 'easy', 'zone', 'reader', 'avanquest', 'partners', 'office', 'hoyle', 'ext', 'maker', 'premier', 'print', 'intuit', 'business', 'subscription', 'laser', 'workshop', 'autodesk', 'year', 'backup', 'sound', 'systems', 'screen', 'big', 'premiere', 'english', 'open', 'quickbooks', 'iii', 'allume', 'educational', 'brain', 'services', 'technical', 'homespun', 'laplink', 'websphere', 'filemaker', 'development', 'antivirus', 'forms', 'jumpstart', 'shop', 'magix', 'users', 'teaches', 'tools', 'adventure', 'kids', 'envelopes', 'corp', 'academic', 'bundle', 'play', 'emedia', 'option', 'mcafee', 'email', 'nova', 'nintendo', 'checks', 'compatible', 'box', 'john', 'master', 'mobile', 'ios', 'essentials', 'symantec', 'card', 'svr', 'ibm', 'piano', 'yamaha', 'creator', 'plus', 'auth', 'protection']
blocks_new = ['corporation', 'game', 'finance', 'audio', 'retrospect', 'full', 'architectural', 'ios', 'entertainment', 'best', 'nt', 'secure', 'simply', 'homespun', 'screen', 'greatest', 'expansion', 'aspyr', 'brain', 'home', 'instant', 'adventures', 'mike', 'tycoon', 'learn', 'mgr', 'mnt', 'art', 'movie', 'protection', 'visual', 'websphere', 'cut', 'smart', 'star', 'piano', 'autodesk', 'family', 'corp', 'suite', 'authoring', 'puzzle', 'hits', 'ext', 'enterprise', 'pro', 'gold', 'premium', 'kids', 'series', 'web', 'leonard', 'ultimate', 'mac', 'product', 'abacus', 'aggregation', 'ae', 'tools', 'digital', 'spanish', 'package', 'total', 'aquarium', 'performa', 'exec', 'landscape', 'vmware', 'win', 'music', 'norton', 'academic', 'year', 'big', 'email', 'book', 'mobile', 'vpn', 'plus', 'tree', 'users', 'nintendo', 'teaches', 'remote', 'educational', 'super', 'guitar', 'basic', 'elementary', 'tlp', 'premiere', 'media', 'laser', 'john', 'desktop', 'checks', 'adobe', 'advantage', 'global', 'advanced', 'open', 'playstation', 'license', 'small', 'system', 'marketing', 'corel', 'project', 'xp', 'technology', 'studio', 'collection', 'office', 'allume', 'professor', 'express', 'quicken', 'laplink', 'option', 'cisco', 'higher', 'wasp', 'compatible', 'backup', 'apple', 'business', 'emc', 'sweeper', 'servers', 'photoshop', 'workshop', 'easy', 'time', 'pianosoft', 'free', 'master', 'single', 'english', 'set', 'upg', 'cs', 'microsoft', 'laptops', 'company', 'play', 'svr', 'ultra', 'sos', 'architect',
              'maintenance', 'insignia', 'flash', 'encore', 'security', 'filemaker', 'network', 'essentials', 'usa', 'emedia', 'linux', 'gatdef', 'truprevent', 'macintosh', 'mcafee', 'sibelius', 'support', 'editing', 'creative', 'watchguard', 'unlimited', 'design', 'storage', 'std', 'premier', 'card', 'technical', 'final', 'acrobat', 'edition', 'sonicwall', 'image', 'ibm', 'freeverse', 'magix', 'years', 'steinberg', 'kit', 'data', 'additional', 'photo', 'limited', 'box', 'sb', 'panda', 'cd', 'vista', 'grade', 'development', 'typing', 'hoyle', 'word', 'platinum', 'windows', 'sage', 'software', 'volume', 'reader', 'subscription', 'guide', 'manager', 'video', 'user', 'professional', 'sony', 'preschool', 'hp', 'auth', 'sound', 'discount', 'games', 'engineering', 'dvd', 'service', 'iii', 'xbox', 'yamaha', 'csdc', 'symantec', 'agent', 'elements', 'animation', 'pack', 'world', 'accounting', 'appl', 'llc', 'jumpstart', 'server', 'version', 'internet', 'antivirus', 'publishing', 'envelopes', 'continuous', 'ipod', 'production', 'inc', 'shop', 'complete', 'retail', 'management', 'standard', 'fogware', 'print', 'quick', 'school', 'match', 'learning', 'creator', 'live', 'individual', 'immersion', 'spy', 'bundle', 'zone', 'intuit', 'power', 'nova', 'adventure', 'avanquest', 'tax', 'webroot', 'partners', 'math', 'deluxe', 'client', 'topics', 'maker', 'forms', 'x', 'interactive', 'services', 'upgrade', 'systems', 'logic', 'pc', 'effects', 'quickbooks', 'punch', 'securid', 'success', 'onone', 'window']

blockify(blocks_new, amazon=r'ass2\task1\amazon.csv',
         google=r'ass2\task1\google.csv')

# blocks_new = generate_blocks(google=r'ass2\task1\google.csv')

# amazon_blocks = pandas.read_csv('amazon_blocks.csv', index_col='block_key')
# google_blocks = pandas.read_csv('google_blocks.csv', index_col='block_key')


pass
