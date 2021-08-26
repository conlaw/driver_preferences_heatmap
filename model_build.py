from os import path
import os
import sys, json, time
import pandas as pd
import numpy as np

# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# Read route data
print('Reading Input Data')
training_routes_path=path.join(BASE_DIR, 'data/model_build_inputs/route_data.json')
with open(training_routes_path, newline='') as in_file:
    route_data = json.load(in_file)

route_array = []
route_stops = []

for route in route_data.keys():
    route_array.append({
        'route': route,
        'station_code': route_data[route]['station_code'],
        'route_score': route_data[route]['route_score']
    })
    
    for stop in route_data[route]['stops'].keys():
        route_stops.append({
            'route': route,
            'stop': stop,
            'type': route_data[route]['stops'][stop]['type'],
            'zone_id': route_data[route]['stops'][stop]['zone_id']
        })
route_df = pd.DataFrame.from_records(route_array)
route_stop_df = (pd.DataFrame.from_records(route_stops)
                 .assign(zone_id = lambda df: df.zone_id.fillna(df.type.replace({'Dropoff':None,
                                                                                 'Station':'Z-Station'})))
                )

## Load actual sequences
training_actual_routes_path=path.join(BASE_DIR, 'data/model_build_inputs/actual_sequences.json')
with open(training_actual_routes_path, newline='') as in_file:
    actual_seq = json.load(in_file)

actual_sequences = []
for route in actual_seq.keys():
    for stop in actual_seq[route]['actual'].keys():
        actual_sequences.append(
            {
                'route': route,
                'stop': stop,
                'order': actual_seq[route]['actual'][stop]
            }
        )
        
actual_sequences_df = pd.DataFrame.from_records(actual_sequences)

#Merge relevant data
sequence_with_zone = (actual_sequences_df
                      .merge(route_stop_df[['route','stop','zone_id']], 
                             on=['route','stop'],
                             how='left')
                      .merge(route_df[['route','station_code','route_score']], 
                             on=['route'],
                             how='left'
                            )
                     )


# Helper functions for generating heatmaps
def genAdjacencyMatrix(sequences, unique_zones):
    seq2step = (sequences
            .merge(sequences.assign(order = lambda df: df.order-1),
                   on=['route','order'])
            [['zone_id_x','zone_id_y']]
            .dropna(how='any')
            .query('zone_id_x != zone_id_y')

    )
    
    seq2step = (seq2step
             .append((pd.DataFrame({'zone_id_x': [x for x in unique_zones if x not in seq2step.zone_id_x.unique()],
                                    'zone_id_y': 'DUMMY'
                                   })
                                    .dropna()))
             .append((pd.DataFrame({'zone_id_y': [x for x in unique_zones if x not in seq2step.zone_id_y.unique()],
                                    'zone_id_x': 'DUMMY'
                                   })
                                    .dropna()))
            )
    
    adj = (pd.crosstab(seq2step['zone_id_x'],seq2step['zone_id_y'])
              .drop('DUMMY',axis=1)
              .drop('DUMMY',axis=0)
           .fillna(0)
             )
    
    return adj

def getWeigtedStationAdjacency(sequences, score_weights, laplace_smooth = 0.01):
    unique_zones = sequences.dropna(subset=['zone_id'],how='any').zone_id.unique()
    
    hist_adj_mat = np.zeros((len(unique_zones),len(unique_zones)))+laplace_smooth
    for score in score_weights.keys():
        hist_adj_mat = hist_adj_mat + score_weights[score]*genAdjacencyMatrix(sequences.query(f'route_score == \'{score}\''), 
                                                                   unique_zones)
        
    return hist_adj_mat.div(hist_adj_mat.sum(axis=1),axis=0)

def writeAllAdjacencyMatrices(sequences, score_weights, path):
    
    for depot in sequences.station_code.unique():
        try:
            print('Writing '+depot)
            adj = getWeigtedStationAdjacency(sequences[sequences.station_code == depot],score_weights)
            adj.to_csv(path+depot+'.csv')
        except:
            print('Issue writing depot')
    
    return
    
# Write adjacency matrices
weights_dict = {
    'High': 1,
    'Medium': 1,
    'Low': 1
}

output_path = path.join(BASE_DIR, 'data/model_build_outputs/heatmap/')
os.mkdir(output_path)

writeAllAdjacencyMatrices(sequence_with_zone, weights_dict,output_path)
print('All done!')