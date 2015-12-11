import requests
import urllib
import json
import os

#http://134.130.232.9:50000/AppParkServer/BrickServlet?orderID=21913&resourceType=json
#http://134.130.232.9:50000/AppParkServer/BrickServlet?filename=brickID21953left.png

order_url_start = 'http://134.130.232.9:50000/AppParkServer/BrickServlet?orderID='
order_url_end = '&resourceType=json'
pic_url_start = 'http://134.130.232.9:50000/AppParkServer/BrickServlet?filename='

r = requests.get('http://134.130.232.9:50000/AppParkServer/OrderServlet?progress=99')
json_request = r.json()
orders = []
bricks = []
# get all bricks
for i in json_request:
	# an order looks like this:
	# {"brickID":21914,"color":15,"hasStuds":1,"offsetX":3,"offsetY":0,"offsetZ":3,"sizeX":4,"sizeZ":2,
	# "images":[{"front":"brickID21914front.png"},{"back":"template303back.png"},
	# {"left":"template303left.png"},{"right":"template303right.png"}]}
	order = requests.get(order_url_start + str(i['orderID']) + order_url_end).json()
	for brick in order:
		bricks.append(brick)
	orders.append(order)

if not os.path.isdir('../images'):
	os.makedirs('../images')

for brick in bricks:
	brick_folder = '../images/' + str(brick['brickID'])
	if not os.path.isdir(brick_folder):
		os.makedirs(brick_folder)
	# TODO remove old folders
	images = brick['images']
	for image in images:
		if 'front' in image:
			urllib.urlretrieve(pic_url_start + image['front'], brick_folder + '/front.png')
		if 'back' in image:
			urllib.urlretrieve(pic_url_start + image['back'], brick_folder + '/back.png')
		if 'left' in image:
			urllib.urlretrieve(pic_url_start + image['left'], brick_folder + '/left.png')
		if 'right' in image:
			urllib.urlretrieve(pic_url_start + image['right'], brick_folder + '/right.png')
