"""A module that uses web-scraping to download sign templates from Wikipedia. By default, it downloads all
sign_templates. Since there are often hundreds, there is an argument (--categories_file) that uses a line separated list
of sign categories to specify the signs to download (by considering the max similarity of the description of any
Wikipedia sign to any category in the categories_file).
"""
# wget for windows needs to be downloaded and added to the PATH variable
# https://eternallybored.org/misc/wget/
import os
import sys
import argparse
import requests
import shutil
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import time
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

parser = argparse.ArgumentParser(description='Downloads sign templates from any Wikipedia or Wikimedia url')
# parser.add_argument('--country', help='The country to download the templates for', default='Germany')
parser.add_argument('--categories_file', help='a text file containing a line seperated list of sign categories to download', default=None)
parser.add_argument('--out_dir', help='the directory to save the templates to', default='./Wiki_Templates')
parser.add_argument('--resolution', help='The resolution for the signs to be downloaded at in px', default='240px')
parser.add_argument('--url', help='Specifies a wiki link', required=True)


class SignTemplate:
    def __init__(self, desc, url, path):
        self.url = url
        self.description = desc
        self.path = path            


def initialise_sign_templates(html_text, resolution, dir):
    soup = BeautifulSoup(html_text, 'html.parser')
    sign_templates = []

    top_level = soup.find("div", class_="mw-parser-output")  # The top level of the html file that contains only the images and text we need
    dir_1 = ""
    dir_2 = ""
    dir_3 = ""
    sign_path = ""
    for elem in top_level.contents:
        if(elem.name == 'h2'):
            # Finds the main section text in the wiki
            dir_1 = "/" + str(elem.find("span", class_="mw-headline".split()).get_text()).replace("/","_").replace(" ","_")
            dir_2 = ""
        elif(elem.name == 'h3'):
            # Finds the subsection text in the wiki
            dir_2 = "/" + str(elem.find("span", class_="mw-headline".split()).get_text()).replace("/","_").replace(" ","_")
            dir_3 = ""
        elif(elem.name == 'h4'):
            # Finds the sub-subsection text
            dir_3 = "/" + str(elem.find("span", class_="mw-headline".split()).get_text()).replace("/","_").replace(" ","_")
        elif(elem.name == 'ul'):
            sign_path = dir + dir_1 + dir_2 + dir_3
            get_templates(sign_templates, elem, resolution, sign_path)
    return sign_templates


def get_templates(sign_templates, soup, resolution, sign_path):
    gallery_boxes = soup.find_all('li', attrs={'class': 'gallerybox'})

    for gbox in gallery_boxes:
        gallerytext = gbox.find('div', attrs={'class': 'gallerytext'}).p
        thumb = gbox.find('div', attrs={'class': 'thumb'}).img
        srcsets = gbox.find('img', attrs = {'srcset' : True})

        if gallerytext is None or thumb is None or srcsets is None:
            continue
        
        # remove alternative language description of sign
        if gallerytext.b:
            gallerytext.b.decompose()

        desc = gallerytext.get_text().strip()
        # url = thumb['srcset'].split(' ')[-2]  # Get the largest image
        url = srcsets['srcset'].split(' ')[-2]
        if not url.startswith(('https:','http:')):  # fixes start of url
            url = 'https:' + url
        url = re.sub("\d+px", resolution, url)  # downloads at specified resolution
        sign_templates.append(SignTemplate(desc, url, sign_path))
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
    headers = {'User-Agent': 'SignScraper/0.0 (kristian.rados@student.curtin.edu.au)'}  # Prevents being denied access
    r = requests.get(args.url, headers=headers)
    # r = requests.get(f'https://en.wikipedia.org/wiki/Road_signs_in_{args.country}', headers=headers) 
    if r.status_code != 200:
        print('Error: {}'.format(r.status_code))
        sys.exit(1)

    # Create a list of sign templates by extracting descriptions and urls of sign images from the HTML
    sign_templates = initialise_sign_templates(r.text, args.resolution, args.out_dir)
    # Filter the list of sign templates by the supplied categories using the TF-IDF algorithm    
    if categories != []:
        sign_templates = filter_templates(categories, sign_templates)
        
    # Download the sign templates
    for i, sign_template in enumerate(sign_templates[:]):
        if categories != []:
            cats_file.write(f'{i + 1}:{categories[i]}\n')
        extension = '.' + sign_template.url.split('.')[-1]
        file_name  = str(i + 1) + extension
        if not os.path.exists(sign_template.path):
            os.makedirs(sign_template.path)
        os.system(f'wget -O {os.path.join(sign_template.path, file_name)} {sign_template.url}')
        time.sleep(0.45)  # Prevents cascading errors which corrupt the images
