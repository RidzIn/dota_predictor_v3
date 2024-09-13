import os
import urllib.request
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random


def generate_data(tournament_html):
    if not os.path.exists(tournament_html):
        raise FileNotFoundError(f"Tournament HTML file not found: {tournament_html}")

    with open(tournament_html, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    tournament_links = []
    for a_tag in soup.find_all("a", href=True, class_="table__body-row"):
        link = a_tag['href']
        if "https://dltv.org/events/" in link:
            tournament_links.append(link)

    headers = {"User-Agent": "Mozilla/5.0"}

    for link in tqdm(tournament_links):
        try:
            request = urllib.request.Request(link, headers=headers)


            time.sleep(random.uniform(1, 2))

            with urllib.request.urlopen(request) as response:
                html = response.read()

            filename = "tournaments/" + link.split("/")[-1] + ".html"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html.decode())

        except urllib.error.HTTPError as e:
            print(f"Error occurred while processing link: {link}")
            print(f"Error message: {str(e)}")


def read_tournaments():
    tournament_dir = "by_year"
    if not os.path.exists(tournament_dir):
        raise FileNotFoundError(f"Tournament directory not found: {tournament_dir}")

    my_files = os.listdir(tournament_dir)
    for filename in my_files:
        try:
            time.sleep(1)
            tournament_path = os.path.join(tournament_dir, filename)
            generate_data(tournament_path)
        except Exception as e:
            print(f"Error occurred while processing file: {filename}")
            print(f"Error message: {str(e)}")


read_tournaments()
