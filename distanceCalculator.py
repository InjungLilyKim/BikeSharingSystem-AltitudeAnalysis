#!/usr/bin/env python

# read latitude and longitutde from csv file
# then calculate the distance and write into new csv file
#
# updated to calculate the bike distance by using MapQuest api
#
# Last update: 2/4/2020
# Author: Injung Kim

import math
import csv
from collections import defaultdict
from pprint import pprint
import json
import requests

MAPQUEST_APP_KEY = "G7LoGyb0mf68nG7IkORMW9U0LOkPDHeG"

def distance_matrix(locations):
    request_body = {
            'locations':
            [{'latLng': {'lat': location['latitude'],
                         'lng': location['longitude']}}
                for location in locations],
            'unit': 'k'
            }
    request_body['routeType'] = 'bicycle'
    r = requests.post('http://open.mapquestapi.com/directions/v2/routematrix?key={appkey}'.format(appkey=MAPQUEST_APP_KEY),
            data=json.dumps(request_body)
            )
    if r.status_code != 200:
        print("We didn't get a response from Mapquest.")
        print("We were trying to access this URL: {0}".format(r.url))
        print("Status code: {0}".format(r.status_code))
        print("Full response headers:")
        pprint(dict(r.headers))
        return
    result = json.loads(r.content)
    try:
        distances = result['distance']
    except KeyError:
        print("We didn't get the response we expected from MapQuest.")
        print("Here's what we got:")
        pprint(result)
        return
    if len(locations) != len(distances):
        print("We didn't get enough distances back for the number of locations.")
        print("Number of locations you supplied: {0}".format(len(locations)))
        print("Number of distances we received: {0}".format(len(distances)))
        return
    # distances are in kilometers, need to convert to meters
    distances = [int(1000*d) for d in distances]
    results = [{'start_id': locations[0]['id'],
                'end_id': locations[loc_index]['id'],
                'distance': distances[loc_index]}
               for loc_index in xrange(len(locations))]
    return results

def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6373.0 # km
    lat1 = math.radians(float(lat1))
    lat2 = math.radians(float(lat2))
    lon1 = math.radians(float(lon1))
    lon2 = math.radians(float(lon2))

    #print(lat1, lat2, lon1, lon2)
    dlat = lat2-lat1
    dlon = lon2-lon1
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(lat1) \
        * math.cos(lat2) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d


# read csv file having latitude and longitude
file1 = open('./latitude_longitude.csv', 'rb')
reader = csv.DictReader(file1)
columns = defaultdict(list)
num = 0 
for row in reader:
 	num = num + 1
	#print(row)
        for (k,v) in row.items():
                columns[k].append(v)

print(columns['station_id'])
print(columns['latitude'])
print(columns['longitude'])
print(num)
file1.close()


# write csv file to have distance info
file2 = open('./bikeDistance.csv', 'wb')
writer = csv.writer(file2)

data = ['']
origin = []
desti = []
distance_data = []
for row in range(num):
	data.append(columns['station_id'][row])
distance_data.append(data)

for i in range(num):
	data = [columns['station_id'][i]]
	for j in range(num):
		origin = [columns['latitude'][i], columns['longitude'][i]]
		desti = [columns['latitude'][j], columns['longitude'][j]]
		dis = distance( origin, desti )
		#dis = str(dis)
		data.append(dis)
	distance_data.append(data)
	        	

writer.writerows(distance_data)
file2.close()

