import urllib2, json, Image, ImageOps
from cStringIO import StringIO

search_term = 'violin'
search_index = 0
n_results = 10
image_results = []


while len(image_results) < n_results:
    search_url = 'http://ajax.googleapis.com/ajax/services/search/images?v=1.0&q=' + search_term + '&start=' + str(search_index)
    search_response = urllib2.urlopen(search_url).read()
    search_results = [result['url'] for result in json.loads(search_response)['responseData']['results']]
    for sr in search_results:
        if len(image_results) > n_results: break;
        try:
            image_data = urllib2.urlopen(sr).read()
            image_results.append(image_data)
        except:
            pass
    search_index += 4

for i,image_data in enumerate(image_results):

    # Write original image
    filename = search_term + '_' + str(i)
    with open('images/original/' + search_term + '/' + filename, 'wb') as f:
        f.write(image_data)

    with open('images/original/' + search_term + '/' + filename, 'r') as f:
        # Create and write resized image
        im = Image.open(StringIO(f.read()))
        resized_image = ImageOps.fit(im, (50,50), Image.ANTIALIAS, centering=(0.5,0.5)).convert('RGB')
        resized_image.save('images/resized/' + search_term + '/' + filename + '.bmp')
