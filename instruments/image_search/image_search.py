import requests, json, Image, ImageOps, sys
from cStringIO import StringIO

if len(sys.argv) < 3:
    print "Usage: %s <search-term> <n-results>" % argv[0]

search_term = sys.argv[1]
n_results = int(sys.argv[2])
search_index = 0
n_retrieved_images = 0

while n_retrieved_images < n_results:

    search_url = 'http://ajax.googleapis.com/ajax/services/search/images?v=1.0&q=' + search_term + '&start=' + str(search_index)
    search_index += 4

    search_request = requests.get(search_url)
    if search_request.status_code != 200:
        print 'bad search request'
        continue
    try:
        search_results = [result['url'] for result in search_request.json()['responseData']['results']]
    except:
        print 'bad search request'
        continue

    for sr in search_results:
        try:
            image_request = requests.get(sr)
        except:
            print 'bad image request'
            continue
        if image_request.status_code != 200:
            print 'bad image request'
            continue
        filename = search_term + str(n_retrieved_images)
        try:
            im = Image.open(StringIO(image_request.content)) 
            resized_image = ImageOps.fit(im, (40,40)).convert('RGB')
            resized_image.save('images/resized/' + search_term + '/' + filename + '.bmp')
        except:
            print 'bad image'
            continue

        n_retrieved_images += 1
        print "%s/%s" % (n_retrieved_images, n_results)

