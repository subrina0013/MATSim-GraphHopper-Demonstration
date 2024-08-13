import numpy as np
import pandas as pd
import os

def gh_output_file():
    gh_edge = pd.read_csv('GraphHopper_DMV\\data\\alledges.csv')
    # print('gh_edge')
    # print(gh_edge.head(1))

    gh_edge_id = pd.read_csv('GraphHopper_DMV\\data\\alledgesID.csv')
    # print('gh_edge_id')
    # print(gh_edge_id.head(1))

    gh_edge_time = pd.read_csv('GraphHopper_DMV\\data\\alledgesTime.csv')
    # print('gh_edge_time')
    # print(gh_edge_time.head(1))

    gh_edge = gh_edge.join(gh_edge_id.edge_id)
    gh_edge = gh_edge.join(gh_edge_time[['time','hr']])
    gh_edge.time = gh_edge.time/1000
    # print('gh_edge')
    # print(gh_edge.head(1))

    gh_osm = pd.read_csv('GraphHopper_DMV\\data\\GH_edges_osmid.csv')
    gh_osm.rename(columns={'edges':'edge_id'}, inplace=True)
    gh_osm.osm_id = gh_osm.osm_id.astype('int64')
    # print('gh_osm')
    # print(gh_osm.head(1))


    gh_osm.loc[gh_osm['type']=='secondary', 'type'] = 'S'
    gh_osm.loc[gh_osm['type']=='primary', 'type'] = 'P'
    gh_osm.loc[gh_osm['type']=='tertiary', 'type'] = 'T'
    gh_osm.loc[gh_osm['type']=='unclassified', 'type'] = 'U'
    gh_osm.loc[gh_osm['type']=='residential', 'type'] = 'R'
    gh_osm.loc[gh_osm['type']=='living_street', 'type'] = 'L'
    gh_osm.loc[gh_osm['type']=='service', 'type'] = 'Sv'
    gh_osm.loc[gh_osm['type']=='track', 'type'] = 'Tra'
    gh_osm.loc[gh_osm['type']=='trunk', 'type'] = 'Tru'
    gh_osm.loc[gh_osm['type']=='motorway', 'type'] = 'M'
    gh_osm.loc[gh_osm['type']=='footway', 'type'] = 'ft'
    gh_osm.loc[gh_osm['type']=='steps', 'type'] = 'st'
    gh_osm.loc[gh_osm['type']=='path', 'type'] = 'pa'
    gh_osm.loc[gh_osm['type']=='pedestrian', 'type'] = 'pe'
    gh_osm.loc[gh_osm['type']=='cycleway', 'type'] = 'cy'
    gh_osm.loc[gh_osm['type']=='bridleway', 'type'] = 'br'
    gh_osm.loc[gh_osm['type']=='corridor', 'type'] = 'co'

    ch_route2 = gh_edge.merge(gh_osm[['edge_id','length','osm_id','type']], on='edge_id', how='left').fillna(0)
    ch_route2 = ch_route2[(ch_route2['type']=='Tru') | (ch_route2['type']=='M') | (ch_route2['type']=='P') | (ch_route2['type']=='S') | (ch_route2['type']=='T') | (ch_route2['type']=='U') | (ch_route2['type']=='R') | (ch_route2['type']=='L')]
    # print('ch_route2')
    # print(ch_route2.head(1))

    ch_time_no_svroad = ch_route2.groupby(['agents','departure_time_sec'])['time'].sum().reset_index()
    ch_time_no_svroad.rename(columns={'time':'GraphHopper Travel Time'}, inplace=True)
    # print('ch_time_no_svroad')
    # print(ch_time_no_svroad.head(1))

    ch_length = ch_route2.groupby(['agents','departure_time_sec'])['distance'].sum().reset_index()
    ch_time_no_svroad = ch_time_no_svroad.merge(ch_length, on=['agents','departure_time_sec'])
    ch_time_no_svroad.rename(columns={'distance':'GraphHopper Distance'}, inplace=True)
    print('ch_time_no_svroad')
    print(ch_time_no_svroad.head(1))

    return ch_route2, ch_time_no_svroad
