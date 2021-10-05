#start of the main file for project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import csv
from pulp import *
import openrouteservice as ors
import folium
from random import randint
def main():
    PlotStores()

    total = []
    tutal = []
    zones = CreateNetwork("AverageDemands.csv", "WoolworthsTravelDurations.csv")
    for zone in zones:
        one, two = CreateNodeSets(zone)
        tone = TrimTours(one)
        ttwo = TrimTours(two)
        total.append(tone)
        tutal.append(ttwo)
    
    WriteToFile(total,tutal)

    rW = LinearProgram("MonFriRoutes.csv", "AverageDemands.csv")
    rS = LinearProgram("SatRoutes.csv", "AverageDemands.csv")

    PlotRoutes(rW)
    return

def PlotRoutes(routes):
    ORSkey = '5b3ce3597851110001cf62485dc6c8ffe33c46e7b4be70ba31980fcb'
    df = pd.read_csv("MonFriRoutes.csv")
    df = df.Route
    locations = pd.read_csv("WoolworthsLocations.csv")
    coords = locations[['Long','Lat']]
    coords  = coords.to_numpy().tolist()
    client = ors.Client(key=ORSkey)
    
    colors = []
    for i in range(len(routes)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    m = folium.Map(location = [-36.95770671222872, 174.81407132219618])
    counter = 0
    for route in routes:
        coords_use = []
        r = route.split("_")
        r2 = df.iloc[int(r[1])]
        r2 = r2.split("--")
        for node in r2:
            for i in range(len(locations)):
                p = locations.iloc[i]
                if p["Store"] == node:
                    coords_use.append(coords[i])
    
        rs = client.directions(coordinates = coords_use, profile = 'driving-hgv', format = 'geojson', validate = False)
        folium.PolyLine(locations = [list(reversed(coord))for coord in rs['features'][0]['geometry']['coordinates']], color = colors[counter]).add_to(m)
        counter += 1
    
    m.save("map.html")
    return

def LinearProgram(routefile, nodefile):
    # read route data
    df1 = pd.read_csv(routefile)
    df2 = pd.read_csv(nodefile)

    # rename routes
    routes_df = pd.Series(df1.Route, index = np.arange(len(df1.Route)))
    # name LP
    prob = LpProblem("WoolworthsRoutingProblem", LpMinimize)
    xt = LpVariable('xt', upBound= 5, lowBound= 0)
    # create variables
    routevars = LpVariable.dicts("Route", routes_df.index, 0, None, LpBinary)
    routes = np.array(routes_df.index)
    c_array = df1.Cost.to_numpy()
    cost = pd.Series(c_array, index = routes)
    # objective function of costs, so divide the time by 4 hours and then multiply by rates
    prob += (lpSum([(routevars[index])*(cost)[index] for index in routes]) + 2000*xt)

    # constraints
    matrix = []
    node_routes = []
    # for each node and each route
    if routefile == "MonFriRoutes.csv":
        for node in df2["Average Demands"]:
            
            for route in df1.Route:
            # if the route contains the node, add to the node_routes array
                if node in route:
                    node_routes.append(1)
                else:
                    node_routes.append(0)
        # reset the node_routes array for the next node
            matrix.append(node_routes)
            node_routes = []
            
    else:
        for node in df2["Average Demands"]:
            if "Countdown" in node:
                if "Metro" not in node:
                    for route in df1.Route:
            # if the route contains the node, add to the node_routes array
                        if node in route:
                            node_routes.append(1)
                        else:
                            node_routes.append(0)
                else:
                    continue
        # reset the node_routes array for the next node
            matrix.append(node_routes)
            node_routes = []
    if routefile == "MonFriRoutes.csv":    
        node_array = df2["Average Demands"].to_numpy()
        nodepatterns = makeDict([node_array, routes], matrix, 0)
    else:
        node_array = []
        for node in df2["Average Demands"]:
            if "Countdown" in node:
                if "Metro" not in node:
                    node_array.append(node)
        nodepatterns = makeDict([node_array, routes], matrix, 0)

    for i in node_array:
        prob += lpSum([routevars[j]*nodepatterns[i][j] for j in routes]) == 1
    # adding 30 truck limit, will need to find out how to do extra cost one
    prob += (lpSum([routevars[j] for j in routes]) - xt) <= 60

    # Solving routines
    if routefile == "SatRoutes.csv":
        prob.writeLP('WoolworthsSat.lp')
    else:
        prob.writeLP('WoolworthsWeek.lp')

    prob.solve()

    print("263 OR Project 2021 \n")

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # The optimised objective function (cost of routing) is printed   
    print("Total Cost of Routes = ", value(prob.objective))


    vars_to_use = []
    for v in prob.variables():
        if v.varValue == 1.0:
            vars_to_use.append(v.name)
            print(v.name, "=", v.varValue)
    return vars_to_use

def WriteToFile(Mon, Sat):
    file = open('MonFriRoutes.csv', 'w', newline= '')
    writer = csv.writer(file)
    header = ["Route", "Cost", "Region"]
    writer.writerow(header)
    for zone in Mon:
        for route in zone:
            string = ""
            time = 0
            for i in range(len(route) - 1):

                string += (route[i].name + "--")
                time += (route[i].dMonFri * 7.5)

                for arc in route[i].arcs_out:
                    if arc.to_store == route[i + 1]:
                        time += (arc.time / 60)
            string += route[-1].name
            cost_ = time/60
            if cost_ <= 4.00:
                cost = 225*cost_
            else:
                extra = cost_ - 4
                cost = extra*275 + ((cost_ - extra)*225)
            writer.writerow([string, str(cost), route[i].region])
    file.close()

    file = open('SatRoutes.csv', 'w', newline= '')
    writer = csv.writer(file)
    header = ["Route", "Cost", "Region"]
    writer.writerow(header)
    for zone in Sat:
        for route in zone:
            string = ""
            time = 0
            for i in range(len(route) - 1):

                string += (route[i].name)
                time += (route[i].dSat * 7.5)

                for arc in route[i].arcs_out:
                    if arc.to_store == route[i + 1]:
                        time += (arc.time / 60)
            string += route[-1].name
            cost_ = time/60
            if cost_ <= 4.00:
                cost = 225*cost_
            else:
                extra = cost_ - 4
                cost = extra*275 + ((cost_ - extra)*225)
            writer.writerow([string, str(cost), route[i].region])
    file.close()
    return

def TrimTours(array):
    trimmed = []
    for tour in array:

        time = 0
        
        for i in range(len(tour) - 1):
            
            time += tour[i].dMonFri * 7.5
            
            for arc in tour[i].arcs_out:
                
                if arc.to_store == tour[i + 1]:
                    
                    time += (arc.time / 60)

        if time < 360:
            trimmed.append(tour)
    return trimmed

def CreateNetwork(filename, travelfile):
    df = pd.read_csv(filename)
    regions_to_read = df.Zone
    regions = []
    for i in range(len(regions_to_read)):
        z = regions_to_read.iloc[i]
        if z not in regions:
            regions.append(z)
    zones = []
    for region in regions:
        test = Network()
        test.read_network(region, filename, travelfile)
        zones.append(test)
    return zones

def PlotStores():
    df = pd.read_csv("WoolworthsLocations.csv")
    BBox = (df.Long.min(),   df.Long.max(),     
         df.Lat.min(), df.Lat.max())
    map = plt.imread("screenshot (126).png")
    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(df.Long[0:55], df.Lat[0:55], zorder=1, alpha = 0.3 ,c='k', s = 20)
    ax.scatter(df.Long[55], df.Lat[55], zorder=1, alpha = 0.3 ,c='r', s = 20)
    ax.scatter(df.Long[56:61], df.Lat[56:61], zorder=1, alpha = 0.3 ,c='m', s = 20)
    ax.scatter(df.Long[61::], df.Lat[61::], zorder=1, alpha = 0.3 ,c='b', s = 20)

    df2 = pd.read_csv("WoolworthsDemands+Average.csv")
    demands = df2["Mon to Fri"].values
    for i in range(65):
        if i > 54:
            plt.annotate(demands[i], (df.Long[i+1], df.Lat[i+1]), size=5)
        else:
            plt.annotate(demands[i], (df.Long[i], df.Lat[i]), size=5)

    ax.set_title('Plotting Stores')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(map, zorder=0, extent = BBox, aspect= 'equal')
    plt.show()

   
    return

def CreateNodeSets(network):
    possible = network.nodes[1::]
    sets = []
    for L in range(1, len(possible)):
        for subset in itertools.combinations(possible, L):
            demand = 0
            for node in subset:
                demand += node.dMonFri
            if demand <= 26:
                sets.append(subset)

    poss_tour = []
    for set in sets:
        start = [network.nodes[0]]
        for node in set:
            start.append(node)
        start.append(network.nodes[0])
        poss_tour.append(start)
        
    sets2 = []
    for L in range(1, len(possible)):
        for subset in itertools.combinations(possible, L):
            sat = True
            for node in subset:
                if (node.dSat == 0):
                    sat = False
            if sat is True:
                demand = 0
                
                for node in subset:
                    demand += node.dSat
                if demand <= 26:
                    sets2.append(subset)

    poss_tour2 = []
    for set in sets2:
        start = [network.nodes[0]]
        for node in set:
            start.append(node)
        start.append(network.nodes[0])
        poss_tour2.append(start)

    return poss_tour, poss_tour2
    
class WoolyStore(object):
    ''' Class for WoolWorths Store
    '''
    def __init__(self, dMonFri, dSat, name, region):
        self.dMonFri = dMonFri # demand for Monday - Friday, will be gotten by csv file
        self.dSat = dSat # same as above
        self.name = name # store will have a name so we know what demand to give etc
        self.region = region
        self.arcs_in = [] # times to self store from other stores
        self.arcs_out = [] # times from self store to other stores
        
    def __repr__(self):
        return "{}".format(self.name)

class Arc(object):
    def __init__(self):
        self.time = None
        self.to_store = None
        self.from_store = None

    def __repr__(self):
        if self.to_store is None:
            to_nd = 'None'
        else:
            to_nd = self.to_store.name
        if self.from_store is None:
            from_nd = 'None'
        else:
            from_nd = self.from_store.name
        return "arc: {}->{}".format(from_nd, to_nd)

class Network(object):
    """ Basic network class.
    """
    def __init__(self):
        self.nodes = []
        self.arcs = []
	
    def __repr__(self):
        return ("ntwk(" + ''.join([len(self.nodes)*'{},'])[:-1]+")").format(*[nd.name for nd in self.nodes])

    def add_node(self, dMonFri, dSat, name, region):
        """ Adds a Node with NAME and VALUE to the network.
        """
        # check node names are unique
        network_names = [nd.name for nd in self.nodes]
        if name in network_names:
            print("Node with name already exists")
		
		# new node, assign values, append to list
        node = WoolyStore(dMonFri, dSat, name, region)
        
        node.dMonFri = dMonFri
        node.dSat = dSat
        node.name = name
        node.region = region

        self.nodes.append(node)

    def join_nodes(self, node_from, node_to, weight):
        """ Adds an Arc joining NODE_FROM to NODE_TO with WEIGHT.
        """
        # new arc
        arc = Arc()
        arc.time = weight
        arc.to_store = node_to
        arc.from_store = node_from
        # append to list
        self.arcs.append(arc)

		# make sure nodes know about arcs
        node_to.arcs_in.append(arc)
        node_from.arcs_out.append(arc)

    def read_network(self, region, demandfile, travelfile):
        """ Read data from FILENAME and construct the network.
        """
        demands = pd.read_csv(demandfile)
        travels = pd.read_csv(travelfile)
        self.add_node(0,0,"Distribution Centre Auckland", "All")
        for i in range(len(demands)):
            p = demands.iloc[i]
            if p["Zone"] == region:
                self.add_node(np.ceil(p["Mon to Fri"]), np.ceil(p["Sat"]), p["Average Demands"], p["Zone"])

        names = travels["Unnamed: 0"]
        for i in range(len(travels)):
            try:
                from_store = self.get_node(names.loc[i])
            except ValueError:
                    continue
            row = travels.loc[i]
            for j in range(len(travels)):
                try:
                    to_store = self.get_node(names.loc[j])
                except ValueError:
                    continue
                time = row[names.loc[j]]
                if(from_store == to_store):
                    continue
                self.join_nodes(from_store, to_store, time)
        
    def get_node(self, name):
        """ Loops through the list of nodes and returns the one with NAME.
		
			Raises NetworkError if node does not exist.
		"""
        for node in self.nodes:
            if node.name == name:
                return node
        raise ValueError("Node does not exist in network")



if __name__ == "__main__":
	 main()