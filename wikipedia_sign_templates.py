import os
import sys
import argparse
import requests
from bs4 import BeautifulSoup

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

parser = argparse.ArgumentParser(description='Downloads sign templates of a particular country from Wikipedia')
parser.add_argument('--country', help='The country to download the templates for', default='Germany')
parser.add_argument('--categories_file', help='a text file containing a line seperated list of sign categories to download', default=None)
parser.add_argument('--out_dir', help='the directory to save the templates to', default='.')


class SignTemplate:
        def __init__(self, desc, url):
            self.url = url
            self.description = desc
            self.type = url.split('.')[-1] # type of image


if __name__ == "__main__":
    args = parser.parse_args()

    # get the list of categories to download
    categories = []
    if args.categories_file is not None:
        with open(args.categories_file, 'r') as f:
            categories = f.read().splitlines()

    # submit a GET request to the Wikipedia page
    r = requests.get(f'https://en.wikipedia.org/wiki/Road_signs_in_{args.country}')
    if r.status_code != 200:
        print('Error: {}'.format(r.status_code))
        sys.exit(1)

    # create a list of sign templates by extracting descriptions and urls of sign images from the HTML
    soup = BeautifulSoup(r.text, 'html.parser')
    gallery_boxes = soup.find_all('li', attrs={'class': 'gallerybox'})
    sign_templates = []

    for gbox in gallery_boxes:
        gallerytext = gbox.find('div', attrs={'class': 'gallerytext'}).p
        thumb = gbox.find('div', attrs={'class': 'thumb'}).a

        if gallerytext is None or thumb is None:
            continue
        
        if gallerytext.b:
            gallerytext.b.decompose()

        desc = gallerytext.get_text().strip()
        url = 'https://en.wikipedia.org' + thumb['href']
        sign_templates.append(SignTemplate(desc, url))
    
    # download the sign templates
    


            





