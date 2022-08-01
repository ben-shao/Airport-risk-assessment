## LIBRARIES

import geopandas
import numpy
import csv
import math
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker, cm
import numpy as np
from numpy import ma
import os

# edit model parameters here
airport_size = 'TwinLakes'                                                                  # <<<<set airport size here, large, medium, small, GA, Mesa12, Luke3
grid_extent = 'CustomGrid'                                                            # <<<<set grid size here, largeGrid, smallGrid, CLGrid (centerline)

write_files = True
runwaylength = 1219                             # in meters
Splitduetowind = [0.5,0.5]                     # Scenario 1 fraction will take off affect the scenario side vs Scenario 2 fraction take off affect the scenario side

# Set up the grid and calculate range
if grid_extent == 'CustomGrid':

    xlim = 10000
    ylim = 4000
    Lgrid = 50


if grid_extent == 'largeGrid':
    xlim = 12000
    ylim = 20000
    Lgrid = 100
    
if grid_extent == 'smallGrid':
    xlim = 12000
    ylim = 5000
    Lgrid = 25
    
if grid_extent == 'CLGrid':
    xlim = 20000
    ylim = 5000  
    Lgrid = 100


                                                        
subdir = 'C:/Users/sam.omelveny/Desktop/Airport Adjacency/Prineville/output/'+airport_size+'/'+grid_extent+'/'

# Create the directory structure
dir1 = 'C:/Users/sam.omelveny/Desktop/Airport Adjacency/Prineville/output/'
if not os.path.exists(dir1):
  os.mkdir(dir1)
dir2 = 'C:/Users/sam.omelveny/Desktop/Airport Adjacency/Prineville/output/'+airport_size+'/'
if not os.path.exists(dir2):
  os.mkdir(dir2)
dir3 = 'C:/Users/sam.omelveny/Desktop/Airport Adjacency/Prineville/output/'+airport_size+'/'+grid_extent+'/'
if not os.path.exists(dir3):
  os.mkdir(dir3)


# number of movements per year for each aircraft category                      # these are half the values for the rep. airport since we are modeling one end of the runway
# LARGE AIRPORT
if airport_size == 'large':
    nm_ac = 406160                                                                  # number of total air carrier movements
    nm_r  = 42324                                                                  # number of total regional movements
    nm_ga = 3551                                                                   # number of total general aviation movements
    nm_m  = 116                                                                   # number of total military movements
    nm_tot = nm_ac + nm_r + nm_ga + nm_m                                           # total number of movements 


# # MEDIUM AIRPORT
if airport_size == 'medium':
    nm_ac = 83577                                                                  # number of total air carrier movements
    nm_r  = 13804                                                                  # number of total regional movements
    nm_ga = 18483                                                                   # number of total general aviation movements
    nm_m  = 1619                                                                   # number of total military movements
    nm_tot = nm_ac + nm_r + nm_ga + nm_m                                           # total number of movements 


# # SMALL AIRPORT
if airport_size == 'small':
    nm_ac = 94579                                                                  # number of total air carrier movements
    nm_r  = 10117                                                                  # number of total regional movements
    nm_ga = 9317                                                                   # number of total general aviation movements
    nm_m  = 714                                                                    # number of total military movements
    nm_tot = nm_ac + nm_r + nm_ga + nm_m                                           # total number of movements                     

# # GENERAL AVIATION AIRPORT
if airport_size == 'TwinLakes':  
    nm_ac = 0                                                                  # number of total air carrier movements
    nm_r  = 0   #25                                                                  # number of total regional movements
    nm_ga = 750  #725                                                                   # number of total general aviation movements
    nm_m  = 0                                                                   # number of total military movements
    nm_tot = nm_ac + nm_r + nm_ga + nm_m   
    
if airport_size == 'GA':  
    nm_ac = 0                                                                  # number of total air carrier movements
    nm_r  = 10937                                                                  # number of total regional movements
    #nm_r = 0
    nm_ga = 46000                                                                   # number of total general aviation movements
    nm_ga = 0
    nm_m  = 1                                                                    # number of total military movements
    nm_tot = nm_ac + nm_r + nm_ga + nm_m   
    
# # MESA AIRPORT                                                                # movements associated with landings on 12 and take offs on 30
if airport_size == 'Mesa12':  
    nm_ac = 6099                                                                  # number of total air carrier movements
    nm_r  = 22662                                                                  # number of total regional movements
    nm_ga = 112768                                                                   # number of total general aviation movements
    nm_m  = 2933                                                                    # number of total military movements
    nm_tot = nm_ac + nm_r + nm_ga + nm_m     

# # LUKE AIRPORT                                                                # movements associated with landings on 3 and take offs on 21
if airport_size == 'Luke3':  
    nm_ac = 0                                                                  # number of total air carrier movements
    nm_r  = 0                                                                  # number of total regional movements
    nm_ga = 0                                                                   # number of total general aviation movements
    nm_m  = 50000                                                                    # number of total military movements
    nm_tot = nm_ac + nm_r + nm_ga + nm_m     

print('No. air carrier movements: ' + str(nm_ac))                                        # total number of movements     


# crash rate per movement for each aircraft category
# from DOE, 1996
cr_ac = 2.35e-7                                                                 # number of air carrier crashes per movement
cr_r  = 1.65e-6                                                                 # number of regional crashes per movement
cr_ga = 1.02e-5                                                                 # number of general aviation crashes per movement
cr_m  = 1.80e-6                                                                 # number of military crashes per movement



# background crash rate for each aircraft category 
# from Arup data analysis and DOE, 1996
bcr_ac = 1.31e-7                                                                # number of crashes per sq km
bcr_r  = 1.41e-6                                                                # 
bcr_ga = 2.87e-5                                                                # 
bcr_m  = 1.54e-6                                                                # 
bcr_h  = 1.47e-6

#bcr_ac = 0                                                               # number of crashes per sq km
#bcr_r  = 0                                                                # 
#bcr_ga = 0                                                                # 
#bcr_m  = 0                                                                # 
#bcr_h  = 0


# impact areas for each aircraft category (km2)
# assumes 100% lethality in this zone
# various sources, see summary of data spreadsheet
ia_ac = 2.98E-2
ia_r  = 5.52E-3
ia_ga = 5.52E-4
ia_m  = 1.33E-2
ia_h  = 5.52E-4                                                                   # assume same as GA 

# building area
DC_area = 0.007182                                                                # km2, based on 266m x 27m 12pod design
#DC_area = 0.0364217                                                                # This is the site substation
#DC_area = 0.002275   # This is the size of small substation

xlist = list(range(int(-xlim/2),int(xlim/2),Lgrid))
#xlist = list(range(xlim,-3275-Lgrid,-1*Lgrid))
ylist = list(range(int(-ylim/2),int(ylim/2),Lgrid))
#ylist = list(range(-ylim,-1*Lgrid,Lgrid)) + list(range(2*Lgrid,ylim+Lgrid,Lgrid))   # hacky way to remove grid points along the centerline 
grid = geopandas.GeoDataFrame()  
    
for scenario_id in range(0,2):
  # set up grid
  
                                           # Right side (larger coordinate side)        
  nrows = len(xlist)
  ncols = len(ylist)
  # print(x)
  # print(y)


  
  row = []
  col = []
  xvalue = []
  yvalue = []
  rvalue = []
  tvalue = []

  # Loop over each single grid points
  for i in range(nrows):
      for j in range(ncols):
          
        row.append(i)
        col.append(j)
        xvalue.append(xlist[i])          # x coordinate for each grid point. Origin is the center of xlim defined
        yvalue.append(ylist[j])          # y coordinate for each grid point. Origin is the center of ylim defined
        
        # Recalculte the x,y coordinate use for the euqation. This is for the model of general aircraft as the coordinate is defined from the end of the run way
        if scenario_id == 0:
      
          x_calc = xlist[i]-runwaylength/2
#          x_calc0 = xlist[i]
          y_calc = ylist[j]                                           # Right side (larger coordinate side)
  
        elif scenario_id == 1:
       
          x_calc = -(xlist[i]+runwaylength/2)
 #         x_calc0 = xlist[i]
          y_calc = ylist[j]                                           # Left side (larger coordinate side)
        
        rvalue.append(math.sqrt(x_calc**2+y_calc**2))
        angle = numpy.degrees(numpy.arctan2(abs(y_calc),abs(x_calc))) 
           
        if x_calc > 0:
            tvalue.append(angle)
        else:
            tvalue.append(180.0-angle)
            
            
  grid['row'] = row
  grid['col'] = col        
  grid['x'] = xvalue
  grid['y'] = yvalue
  grid['r'] = rvalue
  grid['t'] = tvalue


  npts = len(grid['x'])
  #print('No. grid pts: ' + str(npts))
  #print(type(npts))

  fxyt  = numpy.zeros(npts)   # Take off risk for general method
  fxyl  = numpy.zeros(npts)   # Landing risk for general method
  fr_ga = numpy.zeros(npts)   # general aviation model
  # fxyti = numpy.zeros(npts)
  # fxyli = numpy.zeros(npts)

  for i in range(npts):
    xi = grid['x'][i]
    yi = grid['y'][i]
    ri = grid['r'][i]
    ti = grid['t'][i]
    # x1 = xi/1000 + Lgrid/2/1000
    # y1 = yi/1000 - Lgrid/2/1000
    # x2 = xi/1000 - Lgrid/2/1000
    # y2 = yi/1000 + Lgrid/2/1000
    fr_ga[i] = (0.08*math.exp(-1.0*ri/2.5/1000.0) * math.exp(-1.0*ti/60.0))



    if scenario_id == 0:
      
      x_calc = xi-runwaylength/2
      #x_calc = xi
      y_calc = yi                                                                 # Right side (larger coordinate side)
  
    elif scenario_id == 1:
       
      x_calc = -(xi+runwaylength/2)
      #x_calc = xi
      y_calc = yi        
    
    if x_calc >= -600:
        fxyt[i] = ((x_calc/1000.0+0.6)/1.44
                    * math.exp(-1.0*(x_calc/1000.0+0.6)/1.2)
                    * (46.25*math.exp(-0.5*((125.0*y_calc/1000.0)**2.0))/((2.0*math.pi)**0.5)
                    + 0.9635*math.exp(-4.1*abs(y_calc/1000.0))
                    + 0.08*math.exp(-1.0*abs(y_calc/1000.0))))
        # if yi != 0:
        #     fxyti[i] = ((((-ct/bt)*(x2+at+ct)*math.exp(-1*(x2+at)/ct))
        #                 - ((-ct/bt)*(x1+at+ct)*math.exp(-1*(x1+at)/ct)))
        #                 * (dt/125 + (mt/nt)*(math.exp(-1*nt*abs(y1))-math.exp(-1*nt*abs(y2)))
        #                 + (pt/qt)*(math.exp(-1*qt*abs(y1))-math.exp(-1*qt*abs(y2)))))
        

        
        
    if x_calc >= -3275:
        fxyl[i] = ((x_calc/1000.0+3.275)/3.24
                    * math.exp(-1.0*(x_calc/1000.0+3.275)/1.8)
                     * (56.25*math.exp(-0.5*((125.0*y_calc/1000.0)**2.0))/((2.0*math.pi)**0.5)
                     + 0.625*math.exp(-1.0*abs(y_calc/1000.0)/0.4)
                     + 0.005*math.exp(-1.0*abs(y_calc/1000.0)/5.0)))
        
        #fxyl[i] = 0
        # if yi != 0:
        #     fxyli[i] = ((((-cl/bl)*(x2+al+cl)*math.exp(-1*(x2+al)/cl))      # 
        #                 - ((-cl/bl)*(x1+al+cl)*math.exp(-1*(x1+al)/cl)))
        #                 * (dl/125 + (ml/nl)*(math.exp(-1*nl*abs(y1))-math.exp(-1*nl*abs(y2)))
        #                 + (pl/ql)*(math.exp(-1*ql*abs(y1))-math.exp(-1*ql*abs(y2)))))
  if scenario_id == 0:
    grid['fr_ga'] = fr_ga
    grid['fxyt']  = fxyt
    grid['fxyl']  = fxyl   
    
 
    
  else:
    grid['fr_ga'] = grid['fr_ga'] + fr_ga
    grid['fxyt']  = grid['fxyt'] + fxyt
    grid['fxyl']  = grid['fxyl'] + fxyl      
    # grid['fxyti'] = fxyti
    # grid['fxyli'] = fxyli


  if scenario_id == 0:
  
    
    nm_takeoff_ac = nm_ac * Splitduetowind[0]
    nm_land_ac = nm_ac * Splitduetowind[1]
    nm_takeoff_r = nm_r * Splitduetowind[0]
    nm_land_r = nm_r * Splitduetowind[1]
    nm_takeoff_ga = nm_ga * Splitduetowind[0]
    nm_land_ga = nm_ga * Splitduetowind[1]
    nm_takeoff_m = nm_m * Splitduetowind[0]
    nm_land_m = nm_m * Splitduetowind[1]

  elif scenario_id == 1:
    
    grid['fr_ga'] = grid['fr_ga'] + fr_ga
    grid['fxyt']  = grid['fxyt'] + fxyt
    grid['fxyl']  = grid['fxyl'] + fxyl   
    
    nm_takeoff_ac = nm_ac * Splitduetowind[1]
    nm_land_ac = nm_ac * Splitduetowind[0]
    nm_takeoff_r = nm_r * Splitduetowind[1]
    nm_land_r = nm_r * Splitduetowind[0]
    nm_takeoff_ga = nm_ga * Splitduetowind[1]
    nm_land_ga = nm_ga * Splitduetowind[0]
    nm_takeoff_m = nm_m * Splitduetowind[1]
    nm_land_m = nm_m * Splitduetowind[0]
  
# calculate airfield crash risk = crash rate x no. movements x f(x,y)/f(r,theta)
  af_ac = cr_ac*(nm_takeoff_ac)* fxyt + cr_ac*(nm_land_ac)*fxyl
  
 # print(nm_takeoff_ga)
  
  af_r = cr_r*(nm_takeoff_r)* fxyt + cr_r*(nm_land_r)*fxyl
  #af_ga = cr_ga*(nm_takeoff_ga)* fxyt + cr_ga*(nm_land_ga)*fxyl

  # Use the simplified theta r equation for the general aviation and airtaxi
  af_ga = cr_ga*nm_ga*fr_ga
  #af_r  = cr_r*nm_r*fr_ga 


  af_m  = cr_m*(nm_takeoff_m)* fxyt + cr_m*(nm_land_m)*fxyl
  #af_m  = cr_m*nm_m*fr_ga
  af_h  = 0


  if scenario_id == 0:
      
    grid['af_ac']  = af_ac
    grid['af_r']  = af_r
    grid['af_ga']  = af_ga
    grid['af_m']  = af_m
    grid['af_h']  = af_h

  else:
    grid['af_ac']  = grid['af_ac'] + af_ac
    grid['af_r']  = grid['af_r'] + af_r
    grid['af_ga']  = grid['af_ga'] + af_ga
    grid['af_m']  = grid['af_m'] + af_m
    grid['af_h']  = grid['af_h'] + af_h

# calculate total crash risk = airfield crash + background crash. This will be outside the secnario loop as background should only be considered once
tcr_ac = grid['af_ac'] + bcr_ac
tcr_r = grid['af_r'] + bcr_r
tcr_ga = grid['af_ga'] + bcr_ga
tcr_m = grid['af_m'] + bcr_m
tcr_h = grid['af_h'] + bcr_h
tcr = tcr_ac + tcr_r + tcr_ga + tcr_m + tcr_h

# calculate crash risk for typical datacenter
tcr = tcr_ac + tcr_r + tcr_ga + tcr_m + tcr_h                                                           # total crash rate (background and airfield) for all aircraft categories (crashes / yr / km2)
tcr_dc = tcr * DC_area                                                                                  # crash risk at a typical DC (crashes / yr)
tcr_ac_dc = tcr_ac * DC_area                                                                            # crash risk from ac only

grid['tcr_ac']  = tcr_ac
grid['tcr_r']  = tcr_r
grid['tcr_ga']  = tcr_ga
grid['tcr_m']  = tcr_m
grid['tcr_h']  = tcr_h
grid['tcr']  = tcr
grid['tcr_dc']  = tcr_dc
grid['tcr_ac_dc']  = tcr_ac_dc

# calculate individual fatality = tcr * ia
ir_ac = tcr_ac * ia_ac
ir_r = tcr_r * ia_r
ir_ga = tcr_ga * ia_ga
ir_m = tcr_m * ia_m
ir_h = tcr_h * ia_h
ir_total = ir_ac + ir_r + ir_ga + ir_m + ir_h
max_ir = max(ir_total)
min_ir = min(ir_total)

grid['ir_ac']  = ir_ac
grid['ir_r']  = ir_r
grid['ir_ga']  = ir_ga
grid['ir_m']  = ir_m
grid['ir_h']  = ir_h
grid['ir_total']  = ir_total

# print(grid)

output_x      = numpy.zeros((nrows,ncols))
output_y      = numpy.zeros((nrows,ncols))
output_r      = numpy.zeros((nrows,ncols))
output_t      = numpy.zeros((nrows,ncols))
output_fxyt   = numpy.zeros((nrows,ncols))
output_fxyl   = numpy.zeros((nrows,ncols))
output_fr_ga  = numpy.zeros((nrows,ncols))
# output_fxyti = numpy.zeros((nrows,ncols))
# output_fxyli = numpy.zeros((nrows,ncols))
output_af_ac  = numpy.zeros((nrows,ncols))
output_af_r  = numpy.zeros((nrows,ncols))
output_af_ga  = numpy.zeros((nrows,ncols))
output_af_m  = numpy.zeros((nrows,ncols))
output_af_h  = numpy.zeros((nrows,ncols))
output_ir_ac  = numpy.zeros((nrows,ncols))
output_ir_r  = numpy.zeros((nrows,ncols))
output_ir_ga  = numpy.zeros((nrows,ncols))
output_ir_m  = numpy.zeros((nrows,ncols))
output_ir_h  = numpy.zeros((nrows,ncols))
output_ir_total  = numpy.zeros((nrows,ncols))
output_tcr_ac = numpy.zeros((nrows,ncols))
output_tcr_r = numpy.zeros((nrows,ncols))
output_tcr_ga = numpy.zeros((nrows,ncols))
output_tcr_m = numpy.zeros((nrows,ncols))
output_tcr_h = numpy.zeros((nrows,ncols))
output_tcr = numpy.zeros((nrows,ncols))
output_tcr_dc = numpy.zeros((nrows,ncols))
output_tcr_ac_dc = numpy.zeros((nrows,ncols))

for i in range(npts):
    ii = grid['row'][i]
    ji = grid['col'][i]
    output_x[ii,ji]     = grid['x'][i]
    output_y[ii,ji]     = grid['y'][i]
    output_r[ii,ji]     = grid['r'][i]
    output_t[ii,ji]     = grid['t'][i]
    output_fxyt[ii,ji]  = grid['fxyt'][i]
    output_fxyl[ii,ji]  = grid['fxyl'][i]
    output_fr_ga[ii,ji] = grid['fr_ga'][i]
    output_af_ac[ii,ji] = grid['af_ac'][i]
    output_af_r[ii,ji] = grid['af_r'][i]
    output_af_ga[ii,ji] = grid['af_ga'][i]
    output_af_m[ii,ji] = grid['af_m'][i]
    output_af_h[ii,ji] = grid['af_h'][i]
    # output_fxyti[ii,ji] = grid['fxyti'][i]
    # output_fxyli[ii,ji] = grid['fxyli'][i]
    output_ir_ac[ii,ji] = grid['ir_ac'][i]
    output_ir_r[ii,ji] = grid['ir_r'][i]
    output_ir_ga[ii,ji] = grid['ir_ga'][i]
    output_ir_m[ii,ji] = grid['ir_m'][i]
    output_ir_h[ii,ji] = grid['ir_h'][i]
    output_ir_total[ii,ji] = grid['ir_total'][i]
    output_tcr_ac[ii,ji] = grid['tcr_ac'][i]
    output_tcr_r[ii,ji] = grid['tcr_r'][i]
    output_tcr_ga[ii,ji] = grid['tcr_ga'][i]
    output_tcr_m[ii,ji] = grid['tcr_m'][i]
    output_tcr_h[ii,ji] = grid['tcr_h'][i]
    output_tcr[ii,ji] = grid['tcr'][i]
    output_tcr_dc[ii,ji] = grid['tcr_dc'][i]
    output_tcr_ac_dc[ii,ji] = grid['tcr_ac_dc'][i]


## plot IR
X = numpy.transpose(output_x)
Y = numpy.transpose(output_y)
Z = numpy.transpose(output_ir_total)

# plot colormesh
#plt.pcolormesh(X,Y,Z,norm=colors.LogNorm(vmin=0.00000001, vmax=0.1),cmap='jet')
#plt.colorbar()
#plt.savefig(subdir+airport_size+'_IR.png')
#plt.show()

## plot IR contours
levels = MaxNLocator(nbins=5).tick_values(Z.min(), Z.max())
#plt.contourf(X,Y,Z,norm=colors.LogNorm(vmin=0.00000001, vmax=0.1),cmap='jet')
#fig, ax = plt.subplots(figsize=(((xlim/1000)+3.275)/2,((ylim/1000)*2)/2))  # tried to plot so x and y were on same scale, didn't work
fig, ax = plt.subplots()
#cs = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), cmap='jet')
cs = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), cmap='jet', vmin=1e-9, vmax=1e-2)
ax.set_title('Individual Risk; ' + airport_size + ' airport; L = ' +  str(Lgrid) + 'm')
cbar = fig.colorbar(cs)
if write_files == True:
    plt.savefig(subdir+airport_size+'_IRcontours.png')
    

# plot crash risk
X = numpy.transpose(output_x)
Y = numpy.transpose(output_y)
Z = numpy.transpose(output_tcr_dc)

#levels = MaxNLocator(nbins=5).tick_values(Z.min(), Z.max())
#levels = np.linspace(5e-7,5e-6,10)
#levels = np.linspace(1e-7,1e-3,10)
#levels = [1e-7,1e-6,1e-5,1e-4,1e-3]
#levels = [1e-7,5e-7,1e-6,5e-6,1e-5,1e-4,1e-3]
#levels = [1e-7,5e-7,1e-6,5e-6,1e-5,1e-4,1e-3]

levels = [1e-7,1e-6,1e-5,1e-4,5e-4] #,6e-6,8e-6,1e-5,2e-5,3e-5]
#levels = [1e-6, 5e-6,1e-5,1e-4,1e-3,2e-3]
#levels = MaxNLocator(nbins=10).tick_values(1e-8, Z.max())
fig, ax = plt.subplots(figsize=(20, 5))
#cs = ax.contourf(X, Y, Z, levels, cmap='jet')
cs = ax.contourf(X, Y, Z, levels, locator=ticker.LogLocator(), cmap='jet')
#ax.set_title('Crash risk; ' + airport_size + ' airport; L = ' +  str(Lgrid) + 'm')
ax.set_title('Crash risk; ' + airport_size + ' airport; L = ' +  str(Lgrid) + 'm')
ax.set_aspect('equal')
cbar = fig.colorbar(cs)
if write_files == True:
    plt.savefig(subdir+airport_size+'_crashriskcontours.png')

## plot crash risk/IR by category
#data = tcr_ac_dc
#name = "tcr_ac_dc"
#X = numpy.transpose(output_x)
#Y = numpy.transpose(output_y)
#Z = numpy.transpose(data)
#
#levels = MaxNLocator(nbins=5).tick_values(Z.min(), Z.max())
#fig, ax = plt.subplots()
#cs = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), cmap='jet', vmin=1e-9, vmax=1e-2)
#ax.set_title(name + '; ' + airport_size + ' airport; L = ' +  str(Lgrid) + 'm')
#cbar = fig.colorbar(cs)
#if write_files == True:
#    plt.savefig(subdir+airport_size+'_crashriskcontours.png')

###### OUTPUT DATA ######

def write2csv(data, units, filename, col_labels, row_labels):
    with open(filename, 'w', newline='') as csvfile:                            #
        output = csv.writer(csvfile, delimiter=',')                             #
        output.writerow([units]+col_labels)                                     #
        for i,row in enumerate(data):                                           #
            try:
                output.writerow([row_labels[i]] + list(row))                    #
            except:
                output.writerow([row_labels[i]] + [row])                        # 

# subdir = 'csv/output/' ## now defined at the top

if write_files == True:
    write2csv(output_r,'(meters)',subdir+airport_size+'_r.csv',ylist,xlist)
    write2csv(output_t,'(degrees)',subdir+airport_size+'_theta.csv',ylist,xlist)
    write2csv(output_fxyt,'(freq per km2)',subdir+airport_size+'_fxyt.csv',ylist,xlist)
    write2csv(output_fxyl,'(freq per km2)',subdir+airport_size+'_fxyl.csv',ylist,xlist)
    write2csv(output_fr_ga,'(freq per km2)',subdir+airport_size+'_fr_ga.csv',ylist,xlist)
    # write2csv(output_fxyti,'(freq)',subdir+'test_fxyti.csv',ylist,xlist)
    # write2csv(output_fxyli,'(freq)',subdir+'test_fxyli.csv',ylist,xlist)
    write2csv(output_af_ac,'(crashes per km2)',subdir+airport_size+'_af_ac.csv',ylist,xlist)
    write2csv(output_af_r,'(crashes per km2)',subdir+airport_size+'_af_r.csv',ylist,xlist)
    write2csv(output_af_ga,'(crashes per km2)',subdir+airport_size+'_af_ga.csv',ylist,xlist)
    write2csv(output_af_m,'(crashes per km2)',subdir+airport_size+'_af_m.csv',ylist,xlist)
    write2csv(output_af_h,'(crashes per km2)',subdir+airport_size+'_af_h.csv',ylist,xlist)
    write2csv(output_tcr_dc,'(crashes per year)',subdir+airport_size+'_tcr_dc.csv',ylist,xlist)
    write2csv(output_ir_ac,'(fatalities per year)',subdir+airport_size+'_ir_ac.csv',ylist,xlist)
    write2csv(output_ir_r,'(fatalities per year)',subdir+airport_size+'_ir_r.csv',ylist,xlist)
    write2csv(output_ir_ga,'(fatalities per year)',subdir+airport_size+'_ir_ga.csv',ylist,xlist)
    write2csv(output_ir_m,'(fatalities per year)',subdir+airport_size+'_ir_m.csv',ylist,xlist)
    write2csv(output_ir_h,'(fatalities per year)',subdir+airport_size+'_ir_h.csv',ylist,xlist)
    write2csv(output_ir_total,'(fatalities per year)',subdir+airport_size+'_ir_total.csv',ylist,xlist)


# The part below is added to expliclty calculate the actual risk number for a given coordinates.
x = [-3138,-3564,-2814,-2542,-2542]
y = [1464,875,517,517,1563]
#x = [-37,-239,310,640,447]
#y = [-468,-546,-1154,-527,-442]
#x = [936,620,1115,1460]
#y = [-767,-1340,-1650,-1050]

#x=[982]
#y=[888]

#x=[-239,310,620,1115]
#y=[-546,-1154,-1340,-1650]

riskvalue = []
returnperiod = []

for index in range(0,len(x)):
  error = 999999
  for i in range(len(output_x[:,0])):
    if abs(output_x[i][0]-x[index]) < error:
        error = abs(output_x[i][j]-x[index])
        locationx = i
        
  error = 999999
  for j in range(len(output_y[0])):
    if abs(output_y[locationx][j]-y[index]) < error:
        error = abs(output_y[i][j]-y[index])
        locationy = j

  riskvalue.append(Z[locationy][locationx])
  returnperiod.append(1/(Z[locationy][locationx]))



