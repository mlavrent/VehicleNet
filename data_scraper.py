import urllib.request
import urllib
import os


def getLinks(wnid):
    base_url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
    full_url = base_url + wnid

    urllib.request.urlretrieve(full_url, filename="data/links.tmp")
    links = []
    with open("data/links.tmp", 'rb') as link_file:
        for line in link_file:
            links.append(line.strip().decode())

    os.remove("data/links.tmp")
    print(links)
    return links

def download_images(links, folder_name):
    if not os.path.exists("data/" + folder_name):
        os.mkdir("data/" + folder_name)

    i = 0
    for link in links:
        try:
            urllib.request.urlretrieve(link, "data/" + folder_name + "/%05d.png" % i)
            i += 1
        except:
            continue


if __name__ == "__main__":
    links = getLinks("n04166281")
    download_images(links, "sedan")