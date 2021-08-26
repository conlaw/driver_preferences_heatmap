from os import path 
import random
import sys, json, time
import multiprocessing as mp
import pandas as pd
import numpy as np
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

import os

# time we started working 
start_time = time.time()
max_time_allowed = 60*60*3 + 60*55
route_write_number = {0:1000,1:500,2:200,3:50}
max_route_time = 65
# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')

# Read input data
print('Reading Input Data')
# Model Build output
model_build_path=path.join(BASE_DIR, 'data/model_build_outputs/heatmap/')
# Model Apply input
prediction_routes_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json')
with open(prediction_routes_path, newline='') as in_file:
	prediction_routes = json.load(in_file)
travel_times_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json')
with open(travel_times_path, newline='') as in_file:
  travel_times = json.load(in_file)
try:  
	heatmap_files = list(f for f in filter(lambda f: f.endswith('.csv'), os.listdir(model_build_path)  ))
except:
	heatmap_files = []
# csv_lock = mp.Manager().dict({f:mp.Manager().Lock() for f in heatmap_files})
heatmap_dict = mp.Manager().dict({f.split('.')[0] :f for f in heatmap_files})

def propose_all_routes(prediction_routes):
	"""
	Applies `sort_by_key` to each route's set of stops and returns them in a dictionary under `output[route_id]['proposed']`

	EG:

	Input:
	```
	prediction_routes = {
	  "RouteID_001": {
		...
		"stops": {
		  "Depot": {
			"lat": 42.139891,
			"lng": -71.494346,
			"type": "depot",
			"zone_id": null
		  },
		  ...
		}
	  },
	  ...
	}

	print(propose_all_routes(prediction_routes, 'lat'))
	```

	Output:
	```
	{
	  "RouteID_001": {
		"proposed": {
		  "Depot": 0,
		  "StopID_001": 1,
		  "StopID_002": 2
		}
	  },
	  ...
	}
	```
	"""
	proposed_sequences = mp.Manager().dict()
	route_counter  = mp.Value('i',0)
	timedout_routes = []
	# csv_lock = mp.Manager().dict({f:mp.Manager().Lock() for f in heatmap_files})

	# csv_lock = mp.Manager().Lock()

	csv_lock = heatmap_dict

	with ProcessPool(max_workers=mp.cpu_count(),max_tasks =1) as pool:
		future = pool.map(heuristics_tsp, [(p,csv_lock) for p in prediction_routes.keys()], timeout=max_route_time)
		# future = pool.map(heuristics_tsp, prediction_routes.keys(), timeout=max_route_time)

		iterator = future.result()
		while True:
			try:
				result = next(iterator)
			except StopIteration:
				break
			except TimeoutError as error:
				timedout_routes.append(list(prediction_routes.keys())[route_counter.value])
				print("Route timed out - %s" % list(prediction_routes.keys())[route_counter.value])
			except ProcessExpired as error:
				timedout_routes.append(list(prediction_routes.keys())[route_counter.value])
				print("%s. Exit code: %d" % (error, error.exitcode))
			except Exception as error:
				timedout_routes.append(list(prediction_routes.keys())[route_counter.value])
				print("Error raised %s" % error)
				print(error.traceback)  # Python's traceback of remote process

			else:
				proposed_sequences[result[0]] = result[1]
			finally:
				hour = int((time.time()-start_time)/(60*60))
				with route_counter.get_lock():
					route_counter.value += 1
					if not route_counter.value % route_write_number[hour]:
						with open(output_path, 'w') as out_file:
							json.dump(proposed_sequences.copy(), out_file)

	with open(output_path, 'w') as out_file:
		json.dump(proposed_sequences.copy(), out_file)

	while timedout_routes:
		route_counter  = mp.Value('i',0)
		new_timedout_routes = []
		extra_time = int(max_time_allowed - (time.time() - start_time))
		if extra_time < 10: 
			break
		with ProcessPool(1) as pool:
			future = pool.map(heuristics_tsp, [(p,csv_lock) for p in timedout_routes], timeout=int(extra_time/len(timedout_routes)))
			# future = pool.map(heuristics_tsp, timedout_routes, timeout=int(extra_time/len(timedout_routes)))
			iterator = future.result()
			while True:
				try:
					result = next(iterator)
				except StopIteration:
					break
				except TimeoutError as error:
					new_timedout_routes.append(list(prediction_routes.keys())[route_counter.value])
					print("Route timed out")
				except ProcessExpired as error:
					print(list(prediction_routes.keys())[route_counter.value])
					new_timedout_routes.append(list(prediction_routes.keys())[route_counter.value])
					print("%s. Exit code: %d" % (error, error.exitcode))
				except Exception as error:
					new_timedout_routes.append(list(prediction_routes.keys())[route_counter.value])
					print("Error raised %s" % error)
					print(error.traceback)  # Python's traceback of remote process
				else:
					proposed_sequences[result[0]] = result[1]
				finally:
					hour = int((time.time()-start_time)/(60*60))
					with route_counter.get_lock():
						route_counter.value += 1
						if not (route_counter.value+len(proposed_sequences)) % route_write_number[hour]:
							with open(output_path, 'w') as out_file:
								json.dump(proposed_sequences.copy(), out_file)
			timedout_routes = new_timedout_routes

	return proposed_sequences.copy()



def heuristics_tsp(args):

	route_id = args[0]
	csv_lock = args[1]
	route_info = prediction_routes[route_id]
	ttimes = travel_times[route_id]
	try:
		# lock = csv_lock.get(route_info['station_code'], None)
		# lock = csv_lock
		# if lock == None: 
		# 	raise
		# lock.acquire()
		# zone_heatmap = pd.read_csv(model_build_path+route_info['station_code']+'.csv', index_col = 0)
		# lock.release()

		file_name = csv_lock[route_info['station_code']]
		zone_heatmap = pd.read_csv(model_build_path+file_name, index_col = 0)
	except:
		zone_heatmap = pd.DataFrame()

	# zone_heatmap = heatmap_dict.get(route_info['station_code'], pd.DataFrame())

	def min_dist(z1, z2):
		'''Returns the minimum distance between stops in two zones, and the stop in the second zone which achieves
		the minimum.'''
		stops1 = list(zone_df[zone_df['zone_id'] == z1]['stop_id'])[0]
		stops2 = list(zone_df[zone_df['zone_id'] == z2]['stop_id'])[0]
		
		min_time = 9999
		min_s2 = ''
		
		for s1 in stops1:
			for s2 in stops2:
				ttime = ttimes[s1][s2]

				if ttime < min_time:
					min_time = ttime
					min_s2 = s2
		return min_time, min_s2

	def get_zone_distance_df():
		zones = list(zone_df['zone_id'])
		zone_distance_df = pd.DataFrame(0.0, index=zones, columns=zones)

		for index1, row1 in zone_df.iterrows():
			for index2, row2 in zone_df.iterrows():
				z1 = row1['zone_id']
				z2 = row2['zone_id']

				time, _ = min_dist(z1, z2)
				zone_distance_df.at[z1, z2] = 10*time

		return zone_distance_df

	def gen_zone_df():
		'''Generates a DataFrame which partitions stops by zone'''
		stop_df = pd.DataFrame.from_records(list(route_info['stops'].values()))
		stop_df['stop_id'] = list(route_info['stops'].keys())
		
		def name_station_zone(row):
			if row['type'] == 'Station':
				return 'Z-Station'
			else:
				return row['zone_id']
			
		def impute_nan_zone(row):
			if (type(row['zone_id']) == str):
				return row['zone_id']
			else:
				#Find closest zone
				min_dist = 9999
				imputed_zone = 'IMPUTED-ZONE-' + row['stop_id']
				s1 = row['stop_id']
				for s2 in ttimes[s1].keys():
					if (ttimes[s1][s2] < min_dist):
						s2_zone = stop_df[stop_df.stop_id == s2]['zone_id'].iloc[0]
						if type(s2_zone) == str and s2_zone != 'Z-Station':
							min_dist = ttimes[s1][s2]
							imputed_zone = s2_zone
						
				return imputed_zone
			
		def impute_nan_zone_by_merge(row):
			if (type(row['zone_id']) == str) and (row.zone_id in zone_heatmap):
				return row['zone_id']
			else:
				#Find closest zone
				min_dist = 9999
				imputed_zone = 'IMPUTED-ZONE-' + row['stop_id']
				s1 = row['stop_id']
				for s2 in ttimes[s1].keys():
					if (ttimes[s1][s2] < min_dist):
						s2_zone = stop_df[stop_df.stop_id == s2]['zone_id'].iloc[0]
						if type(s2_zone) == str and s2_zone in zone_heatmap and s2_zone != 'Z-Station':
							min_dist = ttimes[s1][s2]
							imputed_zone = s2_zone
						
				return imputed_zone

		stop_df['zone_id'] = stop_df.apply(name_station_zone, axis=1)
		
		zones_in_heatmap = [ i for i in list(stop_df['zone_id'].dropna().unique()) if i in zone_heatmap.index]
		num_known_zones = len(stop_df['zone_id'].dropna().unique())
		
		#Merge zones into heatmap zones if there aren't many (<20% of total zones)
		#print(num_known_zones, len(zones_in_heatmap))
		if num_known_zones - len(zones_in_heatmap) <= 0.2*num_known_zones:
			stop_df['zone_id'] = stop_df.apply(impute_nan_zone_by_merge, axis=1)
		else:
			stop_df['zone_id'] = stop_df.apply(impute_nan_zone, axis=1)
		stop_df['zone_id'] = stop_df.apply(impute_nan_zone, axis=1)
		
		zone_df = (stop_df.groupby('zone_id')
			   .agg({'stop_id': lambda x: list(set(x)), 'type': lambda x: list(x)[0]}
					 ).reset_index()
			   )
		return zone_df


	zone_df = gen_zone_df()
	zone_distance_df = get_zone_distance_df()

	def normalize_heatmap():
		
		'''
		A: ZONES, B:ZONES_TO_EXTRA, C: EXTRA_TO_EXTRA, D:EXTRA_TO_ZONES
		H = A + BD + BCD + ... + BC^nD = A + B(I + C + .. C^n)D = A + B(I - C)^(-1)D
		'''
		
		zones = [ i for i in list(zone_df['zone_id']) if i in zone_heatmap.index]
		extra_zones = [ i for i in zone_heatmap.index if i not in list(zone_df['zone_id'])]

		A = np.array(zone_heatmap.loc[zones,zones])
		B = np.array(zone_heatmap.loc[zones,extra_zones])
		C = np.array(zone_heatmap.loc[extra_zones,extra_zones])
		D = np.array(zone_heatmap.loc[extra_zones,zones])

		H = A + np.dot(np.dot(B,np.linalg.inv(np.subtract(np.identity(len(extra_zones)), C))),D)
		new_heatmap = pd.DataFrame(H, columns = zones, index = zones)
		
		return new_heatmap

	zone_heatmap = normalize_heatmap()
	
	def gen_full_zone_dist_mat():
		zones = list(zone_df['zone_id'])
		distance_mat = pd.DataFrame(0, index=zones, columns=zones)

		for index1, row1 in zone_df.iterrows():
			for index2, row2 in zone_df.iterrows():
				z1 = row1['zone_id']
				z2 = row2['zone_id']

				time, _ = min_dist(z1, z2)
				distance_mat.at[z1, z2] = 10*time

		return np.array(distance_mat), zones

	def gen_zone_dist_mat():
		'''Returns the zone distance matrix. In this implementation, 
			the distance matrix is based on the generated heatmap 
			As the depot has no zone, assign values prop to dist'''
		
		zones = [ i for i in list(zone_df['zone_id']) if i in zone_heatmap.index]
		distance_mat = -np.log(zone_heatmap.loc[zones,zones])*100

		return np.array(distance_mat), zones
	
	def zone_tsp():

		zones_in_heatmap = [ i for i in list(zone_df['zone_id']) if i in zone_heatmap.index]
		num_known_zones = len(list(zone_df['zone_id']))
		
		if num_known_zones - len(zones_in_heatmap) <= 0.5*num_known_zones:
			distance_mat, zones = gen_zone_dist_mat()
		else:
			distance_mat, zones = gen_full_zone_dist_mat()
				
		index_to_zone = dict(zip(list(np.arange(0, (len(zones)))), zones))

		def create_data_model():
			"""Stores the data for the problem."""
			data = {}
			data['distance_matrix'] = distance_mat
			n = data['distance_matrix'].shape[0]
			data['num_vehicles'] = 1
			data['depot'] = n - 1
			return data

		data = create_data_model()

		manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
		routing = pywrapcp.RoutingModel(manager)

		def distance_callback(from_index, to_index):
			"""Returns the distance between the two nodes."""
			# Convert from routing variable Index to distance matrix NodeIndex.
			from_node = manager.IndexToNode(from_index)
			to_node = manager.IndexToNode(to_index)
			return data['distance_matrix'][from_node][to_node]

		transit_callback_index = routing.RegisterTransitCallback(distance_callback)
		routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

		search_parameters = pywrapcp.DefaultRoutingSearchParameters()
		search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

		def get_routes(solution, routing, manager):
			"""Get vehicle routes from a solution and store them in an array."""
			# Get vehicle routes and store them in a two dimensional array whose
			# i,j entry is the jth location visited by vehicle i along its route.
			routes = []
			for route_nbr in range(routing.vehicles()):
				index = routing.Start(route_nbr)
				route = [manager.IndexToNode(index)]
				while not routing.IsEnd(index):
					index = solution.Value(routing.NextVar(index))
					route.append(manager.IndexToNode(index))
				routes.append(route)
			return routes

		solution = routing.SolveWithParameters(search_parameters)
		routes = get_routes(solution, routing, manager)
		zone_ordering = [index_to_zone[index] for index in routes[0][:-1]]
		return zone_ordering, zone_df
	


	zone_ordering, zone_df = zone_tsp()

	def append_zones(z,zone_ordering):
		'''Adds zones for which we had no historical data'''
	
		if z == 'Z-Station':
			min_d = 10000
			index = -1
			for i in zone_ordering:
				dist, _ = zone_distance_df.loc[z,i] 
				if dist < min_d:
					min_d = dist
					index = zone_ordering.index(i)
			return ['Z-Station'] + zone_ordering[index:] + zone_ordering[0:index]
		else:
			min_d = 20000
			index = -1
			for i in range(len(zone_ordering)):
				dist = zone_distance_df.loc[zone_ordering[i],z] \
					+ zone_distance_df.loc[z,zone_ordering[(i+1)%(len(zone_ordering))]]
				if dist < min_d:
					min_d = dist
					index = i
			return zone_ordering[0:i]+[z]+zone_ordering[i:]
	
	def center_depot(zone_ordering):
		for i in zone_ordering:
			if i == 'Z-Station':
				index = zone_ordering.index(i)
				return zone_ordering[index:] + zone_ordering[0:index]
	

	extra_zones = [ i for i in list(zone_df['zone_id']) if i not in zone_ordering]
	
	if 'Z-Station' in extra_zones:
		print('Z-Station not in heat map')
	
	for z in extra_zones:
		zone_ordering = append_zones(z,zone_ordering)

	zone_ordering = center_depot(zone_ordering)

	path_tsp_ends = {}
	for i in range(len(zone_ordering)):
		z1 = zone_ordering[i]
		if (i == len(zone_ordering) - 1):
			path_tsp_ends[z1] = list(zone_df[zone_df['zone_id'] == 'Z-Station']['stop_id'])[0][0]
		else:
			z2 = zone_ordering[i+1]
			_, s2 = min_dist(z1, z2)
			path_tsp_ends[z1] = s2
		
	def min_dist_sz(stop, zone):
		'''Returns the minimum distance between a stop and the closest stop in given zone,
		and the stop in the zone which achieves the minimum.'''
		stops = list(zone_df[zone_df['zone_id'] == zone]['stop_id'])[0]
		
		min_time = 9999
		min_s = ''
		
		for s in stops:
			ttime = ttimes[stop][s]

			if ttime < min_time:
				min_time = ttime
				min_s = s
		return min_time, min_s

	def stop_tsp(zone_id, start, next_zone):
		stops = list(zone_df[zone_df['zone_id'] == zone_id]['stop_id'])[0]
		stops.append('next_zone')
		
		next_zone_stops = list(zone_df[zone_df['zone_id'] == next_zone]['stop_id'])[0]
		if 'next_zone' in next_zone_stops:
			next_zone_stops.remove('next_zone')
			
		index_to_stop = dict(zip(list(np.arange(0, (len(stops)))), stops))
		stop_to_index = dict(zip(stops, list(np.arange(0, (len(stops))))))

		def gen_stop_dist_mat(zone_id):

			distance_mat = pd.DataFrame(0, index=stops, columns=stops)
			for s1 in stops:
				for s2 in stops:
					distance_mat.at[s1, s2] = 10*ttimes[s1][s2]
			return np.array(distance_mat)
		
		def gen_stop_dist_mat_contracted(zone_id):

			distance_mat = pd.DataFrame(0.0, index=stops, columns=stops)
			for s1 in stops[:-1]:
				for s2 in stops[:-1]:
					distance_mat.at[s1, s2] = 10*ttimes[s1][s2]
				distance_mat.at[s1, 'next_zone'] = 10*min([ttimes[s1][z2] for z2 in next_zone_stops])
			return np.array(distance_mat)

		def create_data_model_s():
			"""Stores the data for the problem."""
			data = {}
			#data['distance_matrix'] = gen_stop_dist_mat(zone_id)
			data['distance_matrix'] = gen_stop_dist_mat_contracted(zone_id)
			data['num_vehicles'] = 1
			data['starts'] = [int(stop_to_index[start])]
			data['ends'] = [int(stop_to_index['next_zone'])]
			return data

		data = create_data_model_s()

		manager = (pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['starts'], data['ends']))
		routing = pywrapcp.RoutingModel(manager)

		def distance_callback(from_index, to_index):
			"""Returns the distance between the two nodes."""
			# Convert from routing variable Index to distance matrix NodeIndex.
			from_node = manager.IndexToNode(from_index)
			to_node = manager.IndexToNode(to_index)
			return data['distance_matrix'][from_node][to_node]

		transit_callback_index = routing.RegisterTransitCallback(distance_callback)
		routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

		search_parameters = pywrapcp.DefaultRoutingSearchParameters()
		search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

		def get_routes(solution, routing, manager):
			"""Get vehicle routes from a solution and store them in an array."""
			# Get vehicle routes and store them in a two dimensional array whose
			# i,j entry is the jth location visited by vehicle i along its route.
			routes = []
			for route_nbr in range(routing.vehicles()):
				index = routing.Start(route_nbr)
				route = [manager.IndexToNode(index)]
				while not routing.IsEnd(index):
					index = solution.Value(routing.NextVar(index))
					route.append(manager.IndexToNode(index))
				routes.append(route)
			return routes

		solution = routing.SolveWithParameters(search_parameters)
		routes = get_routes(solution, routing, manager)
		stop_ordering = [index_to_stop[index] for index in routes[0]]
		#return stop_ordering, stop_ordering[-1]
		return stop_ordering, min_dist_sz(stop_ordering[-2],zone_ordering[(
			zone_ordering.index(zone_id)+1)%len(zone_ordering)])[1]

	next_start = list(zone_df[zone_df['zone_id'] == 'Z-Station']['stop_id'])[0][0]
	order = []
	for i in range(len(zone_ordering)):
		if i == len(zone_ordering)-1:
			sequence, next_start = stop_tsp(zone_ordering[i], next_start, zone_ordering[0])
		else:
			sequence, next_start = stop_tsp(zone_ordering[i], next_start, zone_ordering[i+1])
		order = order + sequence[:-1]

	return route_id,{"proposed" : dict(zip(list(order), list(range(len(order)))))}

print('Applying answer with real model...')
output=propose_all_routes(prediction_routes=prediction_routes)


# Write output data
output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')
with open(output_path, 'w') as out_file:
	json.dump(output, out_file)
	print("Success: The '{}' file has been saved".format(output_path))

print('Done! Took {} seconds'.format(time.time()-start_time))
