#start of the main file for project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    PlotStores()
    return

def PlotStores():
    df = pd.read_csv("WoolworthsLocations.csv")
    BBox = (df.Long.min(),   df.Long.max(),     
         df.Lat.min(), df.Lat.max())
    print(BBox)
    map = plt.imread("screenshot (126).png")
    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(df.Long[0:55], df.Lat[0:55], zorder=1, alpha = 0.3 ,c='g', s = 10)
    ax.scatter(df.Long[55], df.Lat[55], zorder=1, alpha = 0.3 ,c='r', s = 10)
    ax.scatter(df.Long[56:61], df.Lat[56:61], zorder=1, alpha = 0.3 ,c='y', s = 10)
    ax.scatter(df.Long[61::], df.Lat[61::], zorder=1, alpha = 0.3 ,c='b', s = 10)


    ax.set_title('Plotting Stores')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(map, zorder=0, extent = BBox, aspect= 'equal')
    plt.show()
    return




if __name__ == "__main__":
	 main()