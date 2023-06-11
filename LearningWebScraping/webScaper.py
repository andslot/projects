from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import pandas as pd

products = [] #List to store name of the product
prices = [] #List to store price of the product
url="https://www.flipkart.com/clothing-and-accessories/topwear/tshirt/men-tshirt/pr?sid=clo,ash,ank,edy&otracker=categorytree&otracker=nmenu_sub_Men_0_T-Shirts"
page = urlopen(url)

soup = bs(page)
for a in soup.findAll('div', attrs={'class':'_1xHGtK _373qXS'}):
    name = a.find('a', href=True, attrs={'class': 'IRpwTa'})
    price = a.find('div', attrs={'class': '_30jeq3'})
    products.append(name.text)
    prices.append(price.text)

df = pd.DataFrame({'Product Name': products, 'Price': prices})
df.to_csv('products.csv', index=False, encoding='utf-8')