import urllib, json, Image, ImageOps, sys
from cStringIO import StringIO

if len(sys.argv) < 3:
    print "Usage: %s <search-term> <n-results>" % argv[0]

search_term = sys.argv[1]
n_results = int(sys.argv[2])
search_index = 0
n_retrieved_images = 49

while n_retrieved_images < n_results:
    search_url = 'http://ajax.googleapis.com/ajax/services/search/images?v=1.0&q=' + search_term + '&start=' + str(search_index)
    print 'search_index: %s' % search_index

    try:
        search_response = urllib.urlopen(search_url).read()
        search_results = [result['url'] for result in json.loads(search_response)['responseData']['results']]
        for sr in search_results:
            try:
                image_data = urllib.urlopen(sr).read()
                filename = search_term + str(n_retrieved_images)
                with open('images/original/' + filename, 'wb') as f:
                    f.write(image_data)
                with open('images/original/' + filename, 'r') as f:
                    # Create and write resized image
                    im = Image.open(StringIO(f.read()))
                    resized_image = ImageOps.fit(im, (50,50), Image.ANTIALIAS, centering=(0.5,0.5)).convert('RGB')
                    resized_image.save('images/resized/' + filename + '.bmp')
                n_retrieved_images += 1
                print "%s/%s" % (n_retrieved_images, n_results)
            except:
                print 'bad result'
                pass
    except:
        pass
    finally:
        search_index += 4

