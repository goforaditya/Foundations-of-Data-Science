# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:34:13 2018

@author: aDITYA
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
                 (26.627423, 75.032065), #HB5
                 (26.624089, 75.028700), #4A5n6
                 (26.624681, 75.026162),] #4A3n4
                 #(26.623383, 75.023437)] #Guest House
    n=len(Locations)
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
                distance_matrix[i,j]=0
            else:
                # Extracting Distance between ith and jth location 
                # Form the response JSON style data received in dmat
                
                distance_km=int(float(str.split(dmat['rows'][i]['elements'][j]['distance']['text'])[0])*1000)
                distance_matrix[i,j] = distance_km
    
    # Finally Array Created
    self.matrix = distance_matrix
    print(self.matrix)
    
  def Distance(self, from_node, to_node):
    return int(self.matrix[from_node][to_node])



def main():
    import numpy as np
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver import routing_enums_pb2
    
    #names=["B4","HB6","HB5","4A5n6","4A3n4","GH"]
    names=["GirlsH","BoysHostel","Math","BioChem"]
    print(names)
    n=len(names)
    
    # Solving Travelling Salesperson Problem
    num_routes = 1    # The number of routes, which is 1 in the TSP.
    # Nodes are indexed from 0 to tsp_size - 1. The depot is the starting node of the route.
    depot = 0
    
    if n > 0:
        routing = pywrapcp.RoutingModel(n, num_routes,depot)
        #search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # Create the distance callback, which takes two arguments (the from and to node indices)
        # and returns the distance between these nodes.
        dist_between_nodes = CreateDistanceCallback()
        dist_callback = dist_between_nodes.Distance
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)    
        
        # Solve, returns a solution if any.
        assignment = routing.Solve()
        if assignment:
            # Solution cost
            print("\n\nTotal Distance: " + str(float(assignment.ObjectiveValue())/1000.0) + "Kilometers")
            # Only One Route Here
            route_number = 0
            index = routing.Start(route_number) # Index of the variable for the starting node
            route = ''
            
            while not routing.IsEnd(index):
                # Convert variable indices to node indices inthe displayed route.
                route += str(names[routing.IndexToNode(index)]) + ' -> '
                index = assignment.Value(routing.NextVar(index))
            route += str(names[routing.IndexToNode(index)])
            print("+=================================+")
            print("|            ROUTE                |")
            print("+=================================+")
            print("\n\n" + route)
            print("\n\n+=================================+")
        else:
            print("No Solution Found")
    else:
        print("Specify an instance greater than 0")
            
if __name__ == '__main__':
    main()
#%%
import numpy as np
Locations = np.array()

#dmat['rows'][0]['elements'][]['distance']['text']
#Locations = [(26.630797, 75.031648),(26.629988, 75.031891),(26.629176, 75.032110),(26.629176, 75.032110),(26.628563, 75.032194),(26.627927, 75.032026),(26.627927, 75.032026),(26.624089, 75.028700),(26.624681, 75.026162),(26.623383, 75.023437)]
