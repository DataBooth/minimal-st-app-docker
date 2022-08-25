from bs4 import BeautifulSoup
import httpx

def get_rating_for_package(package_name="neuraxle"):
    base_url = "https://snyk.io/advisor/python/"
    with httpx.Client() as client:
        page_code = client.get(f"{base_url}{package_name}")
    return BeautifulSoup(page_code.content, features="lxml")

soup = get_rating_for_package()

# printing the page title

print(soup.find("h1").text)
print(soup.find("h3").text)

for item in soup.find_all("span"):
    if "data-v-3f4fee08 data-v-a5505dfc" in item:
        print(item) 
