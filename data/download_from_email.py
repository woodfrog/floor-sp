import os
from bs4 import BeautifulSoup
import argparse
import urllib.request
import sys

import pdb

def get_all_links(html):
    soup = BeautifulSoup(html, 'lxml')
    tables = soup.find_all('table')
    for table in tables:
        links = [a.get('href') for a in table.find_all('a', href=True)]
        if len(links) == 100:
            print('Get the 100 data links')
            return links
    raise ValueError('Cound not find the 100 links, check the input html file')

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def main(args):
    links = get_all_links(open(args.source))
    for idx, link in enumerate(links):
        filename = link[:link.index('?')].split('/')[-1]
        save_path = os.path.join(args.dir, filename)
        if os.path.exists(save_path):
            print('Skip {}, already exists.'.format(filename))
            continue
        print('{}/{}: Downloading file {}'.format(idx+1, len(links), filename))
        urllib.request.urlretrieve(link, save_path, reporthook)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='email.html', help='path to the source html file')
    parser.add_argument('--dir', type=str, default='./public_100/zips', help='path to save the zip files')
    args = parser.parse_args()
    main(args)
