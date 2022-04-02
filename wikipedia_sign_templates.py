"""A module that uses web-scraping to download sign templates from Wikipedia. By default, it downloads all
sign_templates. Since there are often hundreds, there is an argument (--categories_file) that uses a line separated list
of sign categories to specify the signs to download (by considering the max similarity of the description of any
Wikipedia sign to any category in the categories_file).
"""

import os
import sys
import argparse
import requests
import shutil
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

parser = argparse.ArgumentParser(description='Downloads sign templates of a particular country from Wikipedia')
parser.add_argument('--country', help='The country to download the templates for', default='Australia')
parser.add_argument('--categories_file', help='a text file containing a line seperated list of sign categories to download', default=None)
parser.add_argument('--out_dir', help='the directory to save the templates to', default='./Wiki_Templates')


class SignTemplate:
        def __init__(self, desc, url):
            self.url = url
            self.description = desc
            

def initialise_sign_templates(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    gallery_boxes = soup.find_all('li', attrs={'class': 'gallerybox'})
    sign_templates = []

    for gbox in gallery_boxes:
        gallerytext = gbox.find('div', attrs={'class': 'gallerytext'}).p
        thumb = gbox.find('div', attrs={'class': 'thumb'}).img

        if gallerytext is None or thumb is None:
            continue
        
        # remove alternative language description of sign
        if gallerytext.b:
            gallerytext.b.decompose()

        desc = gallerytext.get_text().strip()
        url = thumb['srcset'].split(' ')[-2]  # Get the largest image
        sign_templates.append(SignTemplate(desc, 'https:' + url))
    return sign_templates
            

def filter_templates(categories, sign_templates):
    vectorizer = TfidfVectorizer()
    final_templates = []
    for cat in categories:
        document_list = [s.description for s in sign_templates]
        document_list.insert(0, cat)
        embeddings = vectorizer.fit_transform(document_list)
        scores = cosine_similarity(embeddings[0], embeddings[1:]).flatten()
        final_templates.append(sign_templates[np.argmax(scores)])
    return final_templates


if __name__ == "__main__":
    args = parser.parse_args()
    
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    
    categories = []
    cats_file = None
    
    # Get the list of sign categories to download
    if args.categories_file is not None:
        with open(args.categories_file, 'r') as f:
            categories = f.read().splitlines()
        cats_file = open(os.path.join(args.out_dir, 'categories.txt'), 'w')
    
    # Submit a GET request to the Wikipedia page
    r = requests.get(f'https://en.wikipedia.org/wiki/Road_signs_in_{args.country}')
    if r.status_code != 200:
        print('Error: {}'.format(r.status_code))
        sys.exit(1)

    # Create a list of sign templates by extracting descriptions and urls of sign images from the HTML
    sign_templates = initialise_sign_templates(r.text)
    
    # Filter the list of sign templates by the supplied categories using the TF-IDF algorithm    
    if categories != []:
        sign_templates = filter_templates(categories, sign_templates)
        
    # Download the sign templates
    for i, sign_template in enumerate(sign_templates[:5]):
        if categories != []:
            cats_file.write(f'{i + 1}:{categories[i]}\n')
        extension = '.' + sign_template.url.split('.')[-1]
        file_name  = str(i + 1) + extension
        os.system(f'wget -O {os.path.join(args.out_dir, file_name)} {sign_template.url}')
        
    
    
    


            





