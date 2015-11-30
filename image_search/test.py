import urllib, json, Image
from cStringIO import StringIO

search_term = 'violin'
search_index = 0
n_results = 4
image_results = []


while len(image_results) < n_results:
    search_url = 'http://ajax.googleapis.com/ajax/services/search/images?v=1.0&q=' + search_term + '&start=' + str(search_index)
    search_response = urllib.urlopen(search_url).read()
    search_results = [result['url'] for result in json.loads(search_response)['responseData']['results']]
    for sr in search_results:
        if len(image_results) > n_results: break;
        try:
            image_data = urllib.urlopen(sr).read()
            image_results.append(image_data)
        except:
            pass
    search_index += 4

for i,image_data in enumerate(image_results):
    # Write original image
    filename = search_term + str(i)
    with open('images/original/' + filename, 'wb') as f:
        f.write(image_data)

    Image.open('images/original/violin0')
