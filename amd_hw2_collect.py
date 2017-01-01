# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 20:17:07 2016

@author: Federico
"""


import bs4
import time
import requests
import os
import pandas as pd

def get_sitemap():
    
    ## If the sitemap already exists
    if os.path.isfile('bbc_sitemap.txt'):
        return

    page = None

    for attempt in range(1, 4):
        page = requests.get('http://www.bbc.co.uk/food/sitemap.xml')
        try:
            page.raise_for_status()
            break
        except requests.RequestException:
            time.sleep(attempt*10)

    if not page:
        raise Exception('Failed to get sitemap.xml')

    sitemap = bs4.BeautifulSoup(page.text, 'xml')

    with open('bbc_sitemap.txt', 'w') as f:
        for line in sitemap.find_all('loc'):
            for string in line.stripped_strings:
                if string.startswith('http://www.bbc.co.uk/food/recipes/'):
                    f.write(string + '\n')
                    
def collect():
    with open('bbc_sitemap.txt', 'r') as f:
        for i,line in enumerate(f.readlines()):
            line = line.strip('\n')
            page = requests.get(line)

            try:
                page.raise_for_status()
            except requests.RequestException:
                time.sleep(5)
                page = requests.get(line)

            soup = bs4.BeautifulSoup(page.text, 'lxml')
            dic = {}
            dic['link'] = line.encode("utf-8")
            dic['title'] = soup.find('h1', class_= 'content-title__text').text.encode("utf-8")

            author = soup.find('a', class_= 'chef__link')
            if author:
                dic['author'] = author.text.encode("utf-8")

            prep_time = soup.find('p', class_= 'recipe-metadata__prep-time')
            if prep_time:
                dic['prep-time'] = prep_time.text.encode("utf-8")

            cook_time = soup.find('p', class_= 'recipe-metadata__cook-time')
            if cook_time:
                dic['cook-time'] = cook_time.text.encode("utf-8")

            serving = soup.find('p', class_= 'recipe-metadata__serving')
            if serving:
                dic['serving'] = serving.text.encode("utf-8")

            diet = soup.find('div', class_= 'recipe-metadata__dietary')
            if diet: 
                dic['diet'] = 'Vegetarian'.encode("utf-8")

            dic['ingredients'] = ''
            for li in soup.find_all('li', class_='recipe-ingredients__list-item'):
                dic['ingredients'] += li.text.encode("utf-8") + ' ' 

            dic['method'] = ''
            for p in soup.find_all('p', class_='recipe-method__list-item-text'):
                dic['method'] += p.text.encode("utf-8") + ' '



            if not os.path.exists("doc-hw"):
                os.mkdir("doc-hw", 0777)

            df=pd.DataFrame(dic, index = [0])
            df.to_csv("doc-hw/"+str(i)+".tsv", sep="\t", encoding="utf-8")
            

def main():
    
    get_sitemap()
    collect()

if __name__ == '__main__':
    main()