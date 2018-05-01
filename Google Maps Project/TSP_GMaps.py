# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:34:13 2018

@author: Frndzzz
"""

#%%
# API Key AIzaSyCwd-zvK_TJgOLVEKz-1G-InNHvptMjTfE

class CreateDistanceCallback(object):
  """Create callback to calculate distances between points."""
  def __init__(self):
    """ Creating Array of distances between points."""
    import googlemaps
    import numpy as np
    # Create Google Maps Client API object inorder to invoke distance matrix Method
    gmaps = googlemaps.Client(key='AIzaSyCwd-zvK_TJgOLVEKz-1G-InNHvptMjTfE')
    
    # Setting Locations to input into the distance matrix function 
    
    
    # Locations are in GPS Coordinates form
    Locations = [(26.628563, 75.032194), #B4
                 (26.627927, 75.032026), #HB6
                 (26.627927, 75.032026), #HB5
                 (26.624089, 75.028700), #4A5n6
                 (26.624681, 75.026162), #Admin
                 (26.623383, 75.023437)] #Guest House
    
    # Coressopnding Addresses 
    Names=["B4","HB6","HB5","4A5n6","4A3n4","GH"]
    n=len(Names)
    
    # Downloading the GMaps data into dmat variable
    dmat = gmaps.distance_matrix(Locations,Locations)
    dmat.keys

    # Distance Matrix stores distances in K.M. 
    distance_matrix = np.zeros((n,n),dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i==j:
                # Handling Diagonal Cases Google Maps API Assigns 1 as default value
                # But here we need 0.0
                distance_matrix[i,j]=0.0 
            else:
                # Extracting Distance between ith and jth location 
                # Form the response JSON style data received in dmat
                
                distance_km=int(float(str.split(dmat['rows'][i]['elements'][j]['distance']['text'])[0])*1000)
                distance_matrix[i,j] = distance_km
    
    # Finally Array Created
    self.matrix = distance_matrix
    
  def Distance(self, from_node, to_node):
    return int(self.matrix[from_node][to_node])

#%%
def main():
    import googlemaps
    import numpy as np
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver import routing_enums_pb2
    
    # Create Google Maps Client API object inorder to invoke distance matrix Method
    gmaps = googlemaps.Client(key='AIzaSyCwd-zvK_TJgOLVEKz-1G-InNHvptMjTfE')
    
    # Setting Locations to input into the distance matrix function 
    
    
    # Locations are in GPS Coordinates form
    Locations = [(26.628563, 75.032194), #B4
                 (26.627927, 75.032026), #HB6
                 (26.627927, 75.032026), #HB5
                 (26.624089, 75.028700), #4A5n6
                 (26.624681, 75.026162), #Admin
                 (26.623383, 75.023437)] #Guest House
    
    # Coressopnding Addresses 
    Names=["B4","HB6","HB5","4A5n6","4A3n4","GH"]
    n=len(Names)
    
    # Downloading the GMaps data into dmat variable
    dmat = gmaps.distance_matrix(Locations,Locations)
    dmat.keys
    
    
    # Distance Matrix stores distances in K.M. 
    distance_matrix = np.zeros((n,n),dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i==j:
                # Handling Diagonal Cases Google Maps API Assigns 1 as default value
                # But here we need 0.0
                distance_matrix[i,j]=0.0 
            else:
                # Extracting Distance between ith and jth location 
                # Form the response JSON style data received in dmat
                
                distance_km=int(float(str.split(dmat['rows'][i]['elements'][j]['distance']['text'])[0])*1000)
                distance_matrix[i,j] = distance_km
    # Multiply by 1000 to convert to meteres
    print(distance_matrix)
    # Solving Travelling Salesperson Problem
                
    num_routes = 1    # The number of routes, which is 1 in the TSP.
    # Nodes are indexed from 0 to tsp_size - 1. The depot is the starting node of the route.
    depot = 0
    
    if n > 0:
        routing = pywrapcp.RoutingModel(n, num_routes,depot)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # Create the distance callback, which takes two arguments (the from and to node indices)
        # and returns the distance between these nodes.
        dist_between_nodes = CreateDistanceCallback()
        dist_callback = dist_between_nodes.Distance
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)    
    
if __name__ == '__main__':
    main()
#%%
import numpy as np
Locations = np.array()

#dmat['rows'][0]['elements'][]['distance']['text']
#Locations = [(26.630797, 75.031648),(26.629988, 75.031891),(26.629176, 75.032110),(26.629176, 75.032110),(26.628563, 75.032194),(26.627927, 75.032026),(26.627927, 75.032026),(26.624089, 75.028700),(26.624681, 75.026162),(26.623383, 75.023437)]
