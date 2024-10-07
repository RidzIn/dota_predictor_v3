import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_heroes_attributes():
    with open('data/util/heroes.txt', 'r', encoding='utf-8') as file:
        heroes = [line.strip() for line in file]

    # Process hero names
    heroes_for_parser = [hero.replace(' ', '').lower() for hero in heroes]
    result_list = {}

    # Initialize WebDriver once
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    wait = WebDriverWait(driver, 10)

    try:
        for i, hero_name in tqdm(enumerate(heroes), total=len(heroes)):
            hero_url_name = heroes_for_parser[i]
            url = f'https://www.dota2.com/hero/{hero_url_name}?l=english'
            result = parse_hero_attribute_data(driver, wait, url)
            if result:
                result_list[hero_name] = result
            else:
                print(f"Failed to parse data for {hero_name}")

    finally:
        driver.quit()

    return result_list

def parse_hero_attribute_data(driver, wait, url):
    try:
        driver.get(url)

        # Wait for the role elements to load
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "_3zWGygZT2aKUiOyokg4h1v")))

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        role_elements = soup.find_all(class_="_3zWGygZT2aKUiOyokg4h1v")
        roles_info = {}

        for element in role_elements:
            role_name_element = element.find(class_="_3Fbk3tlFp8wcznxtXIx19W")
            percentage_element = element.find(style=True)

            if role_name_element and percentage_element:
                role_name = role_name_element.text
                style = percentage_element.get('style', '')
                width_value = style.split('width: ')[1].split('%')[0]
                percentage = round(float(width_value) / 100.0, 2)
                roles_info[role_name] = percentage

        attribute_element = soup.find(class_='_3HGWJjSyOjmlUGJTIlMHc_')
        attribute_text = attribute_element.text if attribute_element else ''
        attribute_map = {'Intelligence': 1, 'Strength': 2, 'Agility': 3, 'Universal': 4}
        roles_info['attribute'] = attribute_map.get(attribute_text, 0)

        return roles_info

    except Exception as e:
        print(f"An error occurred while parsing {url}: {e}")
        return None