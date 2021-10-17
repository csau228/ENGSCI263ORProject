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
import seaborn as sns

def main2():
    # reads in demand files for the different situations
    demandFileNames = generateFiles('AverageDemandGreyLynn.xlsx')
    demandFileNames1 = generateFiles('AverageDemandCentral.xlsx')
    demandFileNames2 = generateFiles('AverageDemandManukau.xlsx')
    demandFileNames3 = generateFiles('AverageDemandSouth.xlsx')

    demandFileNamesAll = demandFileNames + demandFileNames1 + demandFileNames2+ demandFileNames3
    # loops over the demand files
    for demandFileName in demandFileNamesAll:
    
        mean = [] # initilialising the arrays
        total = []
        tutal = []
        zones = CreateNetwork(demandFileName, 'WoolworthsTravelDurations.csv') # creating the network

        for zone in zones: # looping over zones within the network
            one, two = CreateNodeSets(zone) # creating node sets
            tone = TrimTours(one) # trimming down to only feasible routes
            ttwo = TrimTours(two)
            total.append(tone)
            tutal.append(ttwo)

        WriteToFile(total,tutal) # write the data to relevant files
    
        rW = LinearProgram("MonFriRoutes.csv", demandFileName) # solving linear program using feasible writes
        rS = LinearProgram("SatRoutes.csv", demandFileName)

def main():
    #PlotStores() # plots the stores on a map
    
    mean = []
    total = []
    tutal = []
    zones = CreateNetwork("AverageDemands.csv", "WoolworthsTravelDurations.csv") # creates the original network
    for zone in zones: # loops over the zones and generates feasible routes
        one, two = CreateNodeSets(zone)
        tone = TrimTours(one)
        ttwo = TrimTours(two)
        total.append(tone)
        tutal.append(ttwo)
    
    WriteToFile(total,tutal) # writes the feasible routes to a file

    rW = LinearProgram("MonFriRoutes.csv", "AverageDemands.csv") # generates linear program results
    rS = LinearProgram("SatRoutes.csv", "AverageDemands.csv")

    PlotRoutesWeek(rW) # plots the routes on OpenStreetMaps
    PlotRoutesSat(rS)
    
    optWeek = [0]*1000 # initialises an array of size 1000 
    optSat = [0]*1000
    np.random.seed(19442)
    for i in range(len(optWeek)): # performs the simulation for Monday to Friday and Saturday
        optWeek[i] = Simulation(rW, "MonFriRoutes.csv", "MonFri_Demands_Distr.csv")
        optSat[i] = Simulation(rS, "SatRoutes.csv", "Sat_Demand_Distr.csv")

    print("Mean of weekday optimal costs = ", np.mean(optWeek)) # collates relevant data
    print("Mean of Saturday optimal costs = ", np.mean(optSat))
    
    
    mean.append(np.mean(optWeek))
    print(np.std(optWeek))
    print(optWeek[int(len(optWeek)*0.025-1)])
    print(optWeek[int(len(optWeek)*0.975-1)])
    print(np.std(optSat))
    print(optSat[int(len(optSat)*0.025-1)])
    print(optSat[int(len(optSat)*0.975-1)])
    PlotSimulations(optWeek) # plots distribution and confidence interval
    PlotSimulations(optSat)

    return

def generateFiles(ExcelFile):
    
    xls = pd.ExcelFile(ExcelFile)   # Read excel file into pd

    suffix = 'TravelDuration' if 'Travel' in ExcelFile else 'Demand'

    header = ['Average Demands', 'Mon to Fri', 'Sat', 'Zone']   # Header to write into demand csv file

    filenames = []

    # Get names of sheets in excel book
    for i in range(len(xls.sheet_names)):
        filename = xls.sheet_names[i].split(" ")
        filenames.append(''.join(filename[:-1])+ suffix +'.csv')

    # Copy each sheet in excel book into a new csv file
    for i in range(len(filenames)):
        
        # Read sheet into dataframe
        df = pd.read_excel(xls, i)

        # Open new csv file to store dataframe above
        file = open(filenames[i], 'w', newline='')
        writer = csv.writer(file)
        
        # Write header into csv file
        if 'Demand' in filenames[i]:
            writer.writerow(header) 
        
        # 
        for j in range(len(df)):
            writer.writerow(df.iloc[j])

        file.close()
    
    return filenames 

def GenerateDemand(values):
    return np.random.choice(values) # selectes the values at random from historical data

def GenerateTime(min, max):
    return np.random.uniform(min, max) # generates time multiplier from uniform distribution

def PlotSimulations(results):
    results.sort() # sorts the results
    plt.hist(results, histtype='stepfilled', alpha=0.5) # plots the results on a histogram
    plt.axvline(results[int(len(results)*0.025-1)], 0, 1, label='~95% Confidence Interval')
    plt.axvline(results[int(len(results)*0.975-1)], 0, 1)
    plt.axvline(np.mean(results), 0, 100, c='darkred', label='Mean')
    plt.title("Simulation of Optimal Costs")
    plt.xlabel("Objective Function (cost)")
    plt.ylabel("Frequency") 
    plt.show()

    # sns.distplot(results, hist=True, 
    #          color = 'darkblue', 
    #          hist_kws={'edgecolor':'black'},
    #          kde_kws={'linewidth': 4}).set_title("Objective Function (cost)")
    # plt.show()

def Simulation(routes, routefile, demands):
    # get all routes, then generate random demands based on the demand profile
    df = pd.read_csv(routefile) # generates dataframes to use
    df2 = pd.read_csv(demands)
    opt = []
    for route in routes: # loops over the optimal routes in routefile
        overdemand = 0 # initialises variables
        demand = []
        r = route.split("_") # using string manipulation and iloc to find relevant route data
        r2 = df.iloc[int(r[1])]
        cost = r2["Cost"]
        time = r2["Time"]
        r2 = r2["Route"].split("--")

        for node in r2: # loops over each node in the route and generates random demands for each store
            if node == "Distribution Centre Auckland":
                continue
            for i in range(len(df2)):
                p = df2.iloc[i]
                if p["Store"] == node:
                    demand.append(GenerateDemand(p.values[1::]))
        if(sum(demand) > 26): # if the demand goes over the trucks capacity we increment the overdemand
            overdemand += 1
        
        # TRAFFIC SIMULATION
        if(routefile == "MonFriRoutes.csv"):
            time = time + time*GenerateTime(0.18,0.65) #from TomTom traffic data generates time multiplier
        else:
            time = time + time*GenerateTime(0.08, 0.31) # multiplier for weekends
        if time > 6:
            overdemand += 1

        elif time > 4 and time <= 6:
            cost += 275*np.ceil(time-4) # adds new costs
        overdemand = overdemand/3 # assume 3 stores can be visted by truck with excess 
        if len(routes) + overdemand > 60:
            cost += 2000 # wet lease trucks added if trucks used greater than 60, 30 for each shift
            opt.append(cost)
        else:
            cost += 225*4*overdemand
            opt.append(cost)
 
    return sum(opt)


def PlotRoutesSat(routes):
    # uses OpenRouteServices to plot and visualise the routes
    ORSkey = '5b3ce3597851110001cf62485dc6c8ffe33c46e7b4be70ba31980fcb'
    df = pd.read_csv("SatRoutes.csv") # gets Saturday routes
    df1 = pd.read_csv("SatRoutes.csv")
    df = df.Route
    # gets relevant data need for visualisation
    locations = pd.read_csv("WoolworthsLocations.csv")
    coords = locations[['Long','Lat']] 
    coords  = coords.to_numpy().tolist()
    client = ors.Client(key=ORSkey)

    
    colors = [] # generates random colours to use on the map
    for i in range(len(routes)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    # initialises the map to use centred on the distribution centre
    m = folium.Map(location = [-36.95770671222872, 174.81407132219618])
    counter = 0
    # loops over each route and splits route into nodes
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
        # plots the route on the map
        rs = client.directions(coordinates = coords_use, profile = 'driving-hgv', format = 'geojson', validate = False)
        folium.PolyLine(locations = [list(reversed(coord))for coord in rs['features'][0]['geometry']['coordinates']], color = colors[counter]).add_to(m)
        counter += 1
    # plots the chains by colour
    for i in range(0, len(coords)): 
        if locations.Type[i] == "Countdown":
            iconCol = "green"
        elif locations.Type[i] == "FreshChoice":
            iconCol = "blue"
        elif locations.Type[i] == "SuperValue":
            iconCol = "red"
        elif locations.Type[i] == "Countdown Metro":
            iconCol = "orange"
        elif locations.Type[i] == "Distribution Centre":
            iconCol = "black"
        folium.Marker(list(reversed(coords[i])), popup = locations.Store[i], icon = folium.Icon(color = iconCol)).add_to(m)
    m.save("Satroute.html") # saves as html
    return

def PlotRoutesWeek(routes):
    # follows the same process as PlotRoutesSat except saves to a different files
    ORSkey = '5b3ce3597851110001cf62485dc6c8ffe33c46e7b4be70ba31980fcb'
    df = pd.read_csv("MonFriRoutes.csv")
    df1 = pd.read_csv("MonFriRoutes.csv")
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

    for i in range(0, len(coords)): 
        if locations.Type[i] == "Countdown":
            iconCol = "green"
        elif locations.Type[i] == "FreshChoice":
            iconCol = "blue"
        elif locations.Type[i] == "SuperValue":
            iconCol = "red"
        elif locations.Type[i] == "Countdown Metro":
            iconCol = "orange"
        elif locations.Type[i] == "Distribution Centre":
            iconCol = "black"
        folium.Marker(list(reversed(coords[i])), popup = locations.Store[i], icon = folium.Icon(color = iconCol)).add_to(m)

    m.save("mapWeekday.html")
    return

def LinearProgram(routefile, demandfile):
    # reads in relevant data
    df1 = pd.read_csv(routefile)
    df2 = pd.read_csv(demandfile)
    # intialises series of routes by index
    routes_df = pd.Series(df1.Route, index=np.arange(len(df1.Route)))
    # initialises the linear program
    prob = LpProblem("WoolworthsRoutingProblem", LpMinimize)
    # creates the extra truck variable which can be 5 at a maximum
    xt = LpVariable('xt', upBound = 5, lowBound = 0)
    # initialises variables for Linear Program
    routevars = LpVariable.dicts("Route", routes_df.index, 0, None, LpBinary)
    # initialises variables for objective function
    routes = np.array(routes_df.index)
    c_array = df1.Cost.to_numpy()
    cost = pd.Series(c_array, index = routes)

    #objective function
    prob += (lpSum([(routevars[index])*(cost)[index]] for index in routes) + 2000*xt)

    #contraints
    matrix = []
    node_routes = []
    # generating a matrix to use to ensure nodes are only visited once
    # if we are using the Monday routes
    if routefile == "MonFriRoutes.csv":
        # loop over nodes
        for node in df2["Average Demands"]:
            # loop over routes
            for route in df1.Route: 
                route2 = route.split('--')
                notvar = False # boolean to see if node is in a route or not
                for node2 in route2:
                    if node == node2:
                        notvar = True 
                    else:
                        continue
                if notvar == True:
                    node_routes.append(1) # appends a 1 if node is in a route and 0 otherwise
                else:
                    node_routes.append(0)
            matrix.append(node_routes)
            node_routes=[]
    else:
        for node in df2["Average Demands"]:
            if "Countdown" in node:
                if "Metro" not in node: # ensures Countdown Metro doesn't get visited on Saturdays
                    for route in df1.Route:
                        # follows same method as above but for Saturday routes
                        route2 = route.split('--')
                        notvar = False
                        for node2 in route2:
                            if node == node2:
                                notvar = True
                            else:
                                continue
                        if notvar == True:
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
        prob += lpSum([routevars[j]*nodepatterns[i][j] for j in routes]) == 1 # ensures nodes are only visited once
    # ensures only 65 routes can be selected, a plan where each route only visits one node is this situation
    prob += (lpSum([routevars[j] for j in routes]) - xt) <= 60

    if routefile == "SatRoutes.csv":
        day = "Sat"
        prob.writeLP('WoolworthsSat.lp') # writes to lp file
    else:
        prob.writeLP('WoolworthsWeek.lp')
        day = "Weekday"
    prob.solve()

    print("263 OR Project 2021 \n")

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # The optimised objective function (cost of routing) is printed   
    print("Total Cost of Routes = ", value(prob.objective))


    # Write optimal cost information into file
    # Uncomment when modelling store closure scenarios
    # with open("OptimalCost.txt", mode="a", encoding="utf-8") as myFile:
    #     myFile.write("Optimal Cost ({}): ".format(day) + str(value(prob.objective)) + "                " + demandfile[:-4] + "Removed" + "\n")

    vars_to_use = []
    for v in prob.variables():
        if v.varValue == 1.0:
            vars_to_use.append(v.name)
            print(v.name, "=", v.varValue)
    return vars_to_use


def WriteToFile(Mon, Sat):
    # writes the routes to a csv file
    file = open('MonFriRoutes.csv', 'w', newline= '')
    writer = csv.writer(file)
    header = ["Route", "Cost", "Region", "Time"] # sets up the headers for csv file
    writer.writerow(header)
    for zone in Mon: # loop over the zones
        for route in zone: # loop over each feasible route
            string = "" # initialise string and time
            time = 0
            for i in range(len(route) - 1):

                string += (route[i].name + "--") # concatenate store names
                time += (route[i].dMonFri * 7.5) # add unloading time

                for arc in route[i].arcs_out: # look at the arcs and try find the time to store
                    if arc.to_store == route[i + 1]:
                        time += (arc.time / 60) # add the time from store to store
            string += route[-1].name
            cost_ = time/60 # put time into hours
            if cost_ <= 4.00:
                cost = 225*cost_ # add the cost 
            else: # if route goes over 4 hours charge of $275 incurred per hour
                extra = cost_ - 4
                cost = extra*275 + ((cost_ - extra)*225)
            writer.writerow([string, str(cost), route[i].region, str(time/60)]) # write the data
    file.close() # close the file

    # do the same for Saturday routes
    file = open('SatRoutes.csv', 'w', newline= '')
    writer = csv.writer(file)
    header = ["Route", "Cost", "Region", "Time"]
    writer.writerow(header)
    for zone in Sat:
        for route in zone:
            string = ""
            time = 0
            for i in range(len(route) - 1):

                string += (route[i].name + '--')
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
            writer.writerow([string, str(cost), route[i].region, str(time/60)])
    file.close()
    return

def TrimTours(array):
    trimmed = [] # initialise the array
    for tour in array:

        time = 0 # initialise time
        
        for i in range(len(tour) - 1):
            
            time += tour[i].dMonFri * 7.5 # add the pallet unloading
            
            for arc in tour[i].arcs_out: # look for store to store time
                
                if arc.to_store == tour[i + 1]:
                    
                    time += (arc.time / 60) # add time to current time

        if time < 360: # if the time is less than 6 hours add it to list
            trimmed.append(tour)
    return trimmed

def CreateNetwork(filename, travelfile):
    df = pd.read_csv(filename) # gets relevant data
    regions_to_read = df.Zone
    regions = [] # initialises the array
    for i in range(len(regions_to_read)): 
        z = regions_to_read.iloc[i]
        if z not in regions:
            regions.append(z) # add the regions to loop over
    zones = []
    for region in regions: # loop over the regions and add the networks to the zones array
        test = Network()
        test.read_network(region, filename, travelfile)
        zones.append(test)
    return zones

def PlotStores():
    # plots the stores
    df = pd.read_csv("WoolworthsLocations.csv")
    BBox = (df.Long.min(),   df.Long.max(),     
         df.Lat.min(), df.Lat.max()) # bounding box based on furthest stores
    map = plt.imread("screenshot (126).png")
    fig, ax = plt.subplots(figsize = (8,7)) # initialise the plot
    ax.scatter(df.Long[0:55], df.Lat[0:55], zorder=1, alpha = 0.3 ,c='k', s = 20)
    ax.scatter(df.Long[55], df.Lat[55], zorder=1, alpha = 0.3 ,c='r', s = 20)
    ax.scatter(df.Long[56:61], df.Lat[56:61], zorder=1, alpha = 0.3 ,c='m', s = 20)
    ax.scatter(df.Long[61::], df.Lat[61::], zorder=1, alpha = 0.3 ,c='b', s = 20)

    df2 = pd.read_csv("WoolworthsDemands+Average.csv")
    demands = df2["Mon to Fri"].values
    for i in range(65): # adds the stores
        if i > 54:
            plt.annotate(demands[i], (df.Long[i+1], df.Lat[i+1]), size=5)
        else:
            plt.annotate(demands[i], (df.Long[i], df.Lat[i]), size=5)

    ax.set_title('Plotting Stores')
    ax.set_xlim(BBox[0],BBox[1]) # sets the limits of the plot
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(map, zorder=0, extent = BBox, aspect= 'equal') # shows the plot
    plt.show()

   
    return

def CreateNodeSets(network):
    # network is nodes in a zone
    possible = network.nodes[1::] # doesn't include distribution centre
    sets = []
    for L in range(1, len(possible)): # loops over all possible lengths 
        for subset in itertools.combinations(possible, L): # creates node sets based on length L
            demand = 0
            for node in subset:
                demand += node.dMonFri
            if demand <= 26: # if the demand goes under or is the capacity of the truck we add it
                sets.append(subset)

    poss_tour = []
    # adds the distribution centre to the node sets to make a tour
    for set in sets:
        start = [network.nodes[0]]
        for node in set:
            start.append(node)
        start.append(network.nodes[0])
        poss_tour.append(start)
        
    sets2 = []
    # creates node sets for Saturday
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
        # read data for store demands and travel duration between stores
        demands = pd.read_csv(demandfile)
        travels = pd.read_csv(travelfile)

        self.add_node(0,0,"Distribution Centre Auckland", "All") # add node for distribution centre
        
        # add store to network if it is in the specified region
        for i in range(len(demands)):
            p = demands.iloc[i]
            if p["Zone"] == region:
                self.add_node(np.ceil(p["Mon to Fri"]), np.ceil(p["Sat"]), p["Average Demands"], p["Zone"])

        names = travels["Unnamed: 0"]
        
        # loop through every node (+ distribution centre) and set each as source store
        for i in range(len(travels)):
            
            # set source store (continue if not in network)
            try:
                from_store = self.get_node(names.loc[i])
            except ValueError:
                    continue
            row = travels.loc[i]
            
            # get travel duration to every other store in network from source store
            for j in range(len(travels)):
                try:
                    to_store = self.get_node(names.loc[j])
                except ValueError:
                    continue
                time = row[names.loc[j]]
                
                # check if source and destination are the same
                if(from_store == to_store):
                    continue
                
                # join stores with arc weighted by travel duration, and add to network
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
	#main2()