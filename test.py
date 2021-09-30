#start of the main file for project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():

    zones = CreateNetwork("TestSet.csv")
    rMF = []
    rS = []
    for zone in zones:
        MonFri, Sat = CheapestInsertion(zone)
        rMF.append(MonFri)
        rS.append(Sat)
    print(1)
    return

def CreateNetwork(filename):
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
        test.read_network(region)
        zones.append(test)
    return zones


def CheapestInsertion(network):

    routes = []
    partial_soln = []
    partial_soln.append(network.nodes[0])
    partial_soln.append(network.nodes[0])

    nodes_used = []
    nodes_used.append(network.nodes[0])
    
    while len(nodes_used) != len(network.nodes):

        min = np.Inf

        for i in range(len(network.nodes)):
            insert = network.nodes[i]
            if (insert not in partial_soln) and (insert not in nodes_used):

                for i in range(len(partial_soln) - 1):
                    # does every interval pair i.e. 0 1, 1 2, 2 3
                    node_i = partial_soln[i]
                    node_j = partial_soln[i + 1]

                    for arc_i in node_i.arcs_out:

                        if arc_i.to_store == insert:

                            time1 = arc_i.time

                            for arc_j in insert.arcs_out:

                                if arc_j.to_store == node_j:

                                    time2 = arc_j.time

                                    if (time1 + time2 < min):

                                        min = time1 + time2
                                        place = i + 1
                                        to_insert = insert
        z = 0
        for node in partial_soln:
            z += node.dMonFri
        if z + (to_insert.dMonFri)> 26:
            partial_soln = [network.nodes[0], network.nodes[0]] # resetting starting condition
            i = 0
        else:
            partial_soln.insert(place, to_insert)
            nodes_used.append(to_insert)
            old = partial_soln.copy()
            routes.append(old)
    

    to_use = []
    for node in network.nodes:
        if node.dSat != 0:
            to_use.append(node)

    routes2 = []
    partial_soln2 = []
    partial_soln2.append(network.nodes[0])
    partial_soln2.append(network.nodes[0])

    nodes_used2 = []
    nodes_used2.append(network.nodes[0])
    
    while len(nodes_used2) != (len(to_use) + 1):

        min = np.Inf

        for i in range(len(to_use)):

            insert = to_use[i]

            if (insert not in partial_soln2) and (insert not in nodes_used2):

                for i in range(len(partial_soln2) - 1):
                    # does every interval pair i.e. 0 1, 1 2, 2 3
                    node_i = partial_soln2[i]
                    node_j = partial_soln2[i + 1]

                    for arc_i in node_i.arcs_out:

                        if arc_i.to_store == insert:

                            time1 = arc_i.time

                            for arc_j in insert.arcs_out:

                                if arc_j.to_store == node_j:

                                    time2 = arc_j.time

                                    if (time1 + time2 < min):

                                        min = time1 + time2
                                        place = i + 1
                                        to_insert = insert
        z = 0
        for node in partial_soln2:
            z += node.dSat
        if z + (to_insert.dSat)> 26:
            partial_soln2= [network.nodes[0], network.nodes[0]] # resetting starting condition
            i = 0
        else:
            partial_soln2.insert(place, to_insert)
            nodes_used2.append(to_insert)
            old = partial_soln2.copy()
            routes2.append(old)

    return routes, routes2
    
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

    def read_network(self, region):
        """ Read data from FILENAME and construct the network.
        """
        demands = pd.read_csv("TestSet.csv")
        travels = pd.read_csv("TestSet_Travel_Duration.csv")
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