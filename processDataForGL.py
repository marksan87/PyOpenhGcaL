#!/usr/bin/env python
import numpy as np
import pandas as pd

import gzip
import pickle
numlayers = {3:28,4:12} # Subdet:numlayers (1 to value)
# Layers numbered from 1 for both subdetectors
geomFile = "data/cell_map.pklz" 
with gzip.open(geomFile, "rb") as f:
    _cell_map = pickle.load(f)
    _tc_map = pickle.load(f)

#    branches = ["tc_x", "tc_y", "tc_z"]
#    cell_map = _cell_map.query("tc_layer==%d and tc_zside == 1 and tc_subdet == 3" % layer)[branches]

    #tc_branches = ["x","y","z",'triggercell','neighbor_zside', 'neighbor_subdet','neighbor_layer','neighbor_wafer','neighbor_cell','neighbor_distance']
    #tc_map = _tc_map.query("layer == %d and zside == 1 and subdet == 3" % layer)[tc_branches]
tc_branches = ["x","y","z",'triggercell']

tc_map = _tc_map


# Layers for subdet 4 go from 29-40
dataFile = "data/relvalQCD_df.pklz"
with gzip.open(dataFile, "rb") as f:
    df_perCell = pickle.load(f)
    df_perRoc = pickle.load(f)
    df_perWafer = pickle.load(f)
    df_gen = pickle.load(f)
    df_tower = pickle.load(f)


zside = 1
hexagons = {}

for subdet,nlayers in sorted(numlayers.items()):
#    print "Hexagons" 
#    print "subdet = %d\tnlayers = %d" % (subdet,nlayers)
    
    wafer_x = {}
    wafer_y = {}
    wafer_z = {}
    hexagons[subdet] = {}
    for l in xrange(1,nlayers+1):
	tc = tc_map.loc[zside,subdet,l][["x","y","z",'triggercell']]
	allwafers = tc.index.drop_duplicates().values
	hexagons[subdet][l] = []
	for w in allwafers:
	    wafer_x[w],wafer_y[w],wafer_z[w] = tc.loc[w]["x"], tc.loc[w]["y"], tc.loc[w]["z"]
	    hexagons[subdet][l].append( [wafer_x[w].mean(), wafer_y[w].mean(), wafer_z[w].mean(), w] )


waferColors = {}
for evt in xrange(10):
    print "-------Event %d-------" % evt 
    waferColors[evt] = {}
    for s,maxl in sorted(numlayers.items()):
    #for s,maxl in [(3,28)]: #[(3,28),(4,12)]: 
        print "s = %d\tmaxl = %d" % (s,maxl)
        waferColors[evt][s] = {}
        layermin = 1 if s == 3 else numlayers[3]+1
        layermax = numlayers[3] if s == 3 else numlayers[3]+numlayers[4]
        for l in xrange(layermin,layermax+1):
            waferColors[evt][s][l] = {}

            df = df_perWafer.query("event == %d and tc_subdet == %d and tc_layer == %d" % (evt,s,l) )[["tc_wafer","tc_energy"]]
            energyMax = df["tc_energy"].max()
            hexlayer = l if s == 3 else (l - numlayers[s-1]) 
            print "Now on subdet %d  layer %d" % (s,hexlayer)
            for i,h in enumerate(hexagons[s][hexlayer]):
                # Fill colors
                wafer = h[3]
                try:
                    energy = df.query("tc_wafer==%d" % wafer)["tc_energy"].values[0]
                    waferColors[evt][s][l][wafer] = [energy/energyMax, 0.1, 0.1, 1.0]
                except IndexError:
                    # Wafer not found, set default color 
                    waferColors[evt][s][l][wafer] = [0.05,0.75,0.75,0.2]


with gzip.open("hgcalDataForGL.pklz","wb") as f:
    pickle.dump(hexagons,f,protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(waferColors,f,protocol=pickle.HIGHEST_PROTOCOL)
