from bing_image_downloader import downloader
file = open('food.txt', 'r')
lines = file.readlines()
for line in lines:
    str = line.strip()
    downloader.download(str, limit=200)
