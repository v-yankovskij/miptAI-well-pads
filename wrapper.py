import pandas as pd
import json
import numpy as np
import json
import collections
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from copy import copy


def loadWells(wells_df):
    names = [n for n in wells_df['Name']]
    times = [datetime.strptime(date, '%d.%m.%Y') for date in wells_df['Date'].to_numpy(dtype='str')]
    X1 = wells_df['East T1'].to_numpy(dtype=float)
    Y1 = wells_df['North T1'].to_numpy(dtype=float)
    Z1 = wells_df['TVD T1'].to_numpy(dtype=float)
    X3 = wells_df['East T3'].to_numpy(dtype=float)
    Y3 = wells_df['North T3'].to_numpy(dtype=float)
    Z3 = wells_df['TVD T3'].to_numpy(dtype=float)
    return names,times,X1,Y1,Z1,X3,Y3,Z3

def loadDEM(map_df):
    xDem = map_df["X"]
    yDem = map_df["Y"]
    costDem = map_df["Koeff"]
    return xDem, yDem, costDem

def loadAnswer(answer_json):
    xWellPad = []
    yWellPad = []
    wellNames = []
    for well_pad in answer_json:
        xWellPad.append(answer_json[well_pad]["well_pad_x"])
        yWellPad.append(answer_json[well_pad]["well_pad_y"])
        wellNames.append(answer_json[well_pad]["wells"])
    return xWellPad, yWellPad, wellNames
    
class Answer:
    def __init__(self):
        self.well_pads = {}
        self.export = {}
        self.coords = []
    def add_single_well(self, well_pad_x, well_pad_y, name):
        key = (well_pad_x, well_pad_y)
        index = len(self.coords)
        if (key not in self.coords):
            self.coords.append(key)
            self.export["well_pad_"+str(index)] = {}
            self.export["well_pad_"+str(index)]["well_pad_x"] = key[0]
            self.export["well_pad_"+str(index)]["well_pad_y"] = key[1]
            self.export["well_pad_"+str(index)]["wells"] = []
        self.export["well_pad_"+str(self.coords.index(key))]["wells"].append(name)
    def save(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.export, fp,  indent=4)
            

class Well:
    def __init__(self, WellPad, name, t, x1, y1, z1, x3, y3, z3, c3, CONST_L, CONST_PENALTY):
        self.name = name
        self.log = ''
        self.x1 = x1
        self.y1 = y1
        self.z1 = -z1
        self.x3 = x3
        self.y3 = y3
        self.z3 = -z3
        self.t = t
        self.penalty = CONST_PENALTY
        self.x0 = WellPad.x
        self.y0 = WellPad.y
        self.z0 = 0
        '''
        #for circle trajectories
        self.R, self.xc, self.yc, self.zc = self.calculateR()
        self.theta = self.calculateTheta()
        self.L = self.theta * self.R
        '''
        #for spider trajectories
        self.L = np.sqrt((self.x0-self.x1)**2 + (self.y0-self.y1)**2 + (self.z0-self.z1)**2) + np.sqrt((self.x3-self.x1)**2 + (self.y3-self.y1)**2 + (self.z3-self.z1)**2)
        self.c3 = c3
        self.price = self.c3 * self.L
        self.trajectory = self.calculate_spider_trajectory()
        self.const_l = CONST_L
        WellPad.add_well(self)

    def update(self, WellPad):
        self.x0 = WellPad.x
        self.y0 = WellPad.y
        self.z0 = 0
        '''
        #for circle trajectories
        self.R, self.xc, self.yc, self.zc = self.calculateR()
        self.theta = self.calculateTheta()
        self.L = self.theta * self.R
        '''
        #for spider trajectories
        self.L = np.sqrt((self.x0-self.x1)**2 + (self.y0-self.y1)**2 + (self.z0-self.z1)**2) + np.sqrt((self.x3-self.x1)**2 + (self.y3-self.y1)**2 + (self.z3-self.z1)**2)
        self.price = self.c3 * self.L
        self.trajectory = self.calculate_spider_trajectory()

    def calculateR(self):
        A = np.array([self.x0, self.y0, self.z0])
        B = np.array([self.x1, self.y1, self.z1])
        C = np.array([self.x3, self.y3, self.z3])
        a = np.linalg.norm(C - B)
        b = np.linalg.norm(C - A)
        c = np.linalg.norm(B - A)
        s = (a + b + c) / 2
        R = a * b * c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
        b1 = a*a * (b*b + c*c - a*a)
        b2 = b*b * (a*a + c*c - b*b)
        b3 = c*c * (a*a + b*b - c*c)
        center = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
        center /= b1 + b2 + b3
        self.R = R
        self.xc = center[0]
        self.yc = center[1]
        self.zc = center[2]
        return R, center[0], center[1], center[2]
    
    def calculate_circle_trajectory(self):
        x_axis = np.array([self.x0-self.xc, self.y0-self.yc, self.z0-self.zc])
        tmp = np.array([self.x1-self.x3, self.y1-self.y3, self.z1-self.z3])
        z_axis = np.cross(x_axis, tmp)
        y_axis = -np.cross(x_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)
        phi = np.linspace(0, self.theta, num=10)
        Global_basis = np.array([x_axis, y_axis, z_axis])
        trajectory = []
        for angle in phi:
            r = np.array([self.R * np.cos(angle), self.R * np.sin(angle), 0])
            tmp2 = np.linalg.solve(Global_basis, r)
            trajectory.append(np.array([self.xc, self.yc, self.zc]) + tmp2)
        self.trajectory = np.array(trajectory)
        return np.array(trajectory)
        
    def calculate_spider_trajectory(self):
        trajectory = [[self.x0, self.y0, self.z0], [self.x1, self.y1, self.z1], [self.x3, self.y3, self.z3]]
        self.trajectory = np.array(trajectory)
        return np.array(trajectory)
    
    def calculateTheta(self):
        #calculate arc length in radians
        d2 = (self.x0-self.x3)**2 + (self.y0-self.y3)**2 + (self.z0-self.z3)**2
        if (abs(d2/(2*self.R**2)-2) < 1e-13):
            theta = np.arccos(-1)
        else:      
            theta = np.arccos(1-d2/(2*self.R**2)) 
        if (d2 < (self.x0-self.x1)**2 + (self.y0-self.y1)**2 + (self.z0-self.z1)**2 + (self.x3-self.x1)**2 + (self.y3-self.y1)**2 + (self.z3-self.z1)**2):
            theta = 2*np.pi - theta
        return theta
    
    def check_length(self):
        if (self.L > self.const_l):
            self.log += 'Error: Too long well ' + str(self.name) + '\n'
            self.price += self.penalty * (self.L-self.const_l)  #penalty
            return False
        else:
            return True
    
class WellPad:
    def __init__(self, x, y, alpha, CONST_PENALTY):
        self.log = ''
        self.x = x
        self.y = y
        self.alpha = alpha
        self.wells = {}
        self.penalty = CONST_PENALTY
        self.a = 10
        self.x_min = self.x - 0.5 * self.a
        self.y_min = self.y - 0.5 * self.a
        self.x_max = self.x + 0.5 * self.a
        self.y_max = self.y + 0.5 * self.a
        self.k = 0
       
    def calculateK(self, xDem, yDem, costDem):
        counter = 0
        forbidden = 0
        i_xmin = xDem.searchsorted(self.x_min)
        i_xmax = xDem.searchsorted(self.x_max)
        for i in range(i_xmin, i_xmax): #fast method
            if (self.x_min <= xDem[i] and xDem[i] <= self.x_max and self.y_min <= yDem[i] and yDem[i] <= self.y_max):
                self.k += costDem[i]
                if (costDem[i] >= self.penalty):
                    self.log += 'Error: Well Pad (' + str(self.x) + ',' + str(self.y) + ') is in forbidden zone \n'
                    forbidden = 1  #penalty
                counter += 1
        if (counter > 0):
            self.k /= counter  
        else:
            self.k = 1
        if (forbidden):
            return False
        return True

    def update(self):
        self.x_min = self.x - 0.5 * self.a
        self.y_min = self.y - 0.5 * self.a
        self.x_max = self.x + 0.5 * self.a
        self.y_max = self.y + 0.5 * self.a
        for well_name in self.wells.keys():
            self.wells[well_name].update(self)
                
            
    def add_well(self, Well):
        self.wells[Well.name] = Well
        self.a = len(self.wells) * self.alpha
        self.update()
        
    def check_position(self, x_min, y_min, x_max, y_max):
        if (self.x_min < x_min or self.x_max > x_max or self.y_min < y_min or self.y_max > y_max):
            self.log += 'Error: Well Pad (' + str(self.x) + ',' + str(self.y) + ') is out of bounds \n'
            self.k = 1e5 * self.penalty #penalty
            return False
        else:
            return True
 



 
class Solution:
    def __init__(self, dem_df, wells_df, answer_json, CONST_ALPHA, CONST_C0, CONST_C1, CONST_C2, CONST_C3, CONST_L, CONST_D, CONST_PENALTY):
        self.answer_json = answer_json
        self.Y = self.get_axis(dem_df["Y"])[::-1]
        self.X = self.get_axis(dem_df["X"])
        self.K = dem_df["Koeff"].to_numpy().reshape(self.get_grid_size(dem_df)).T[::-1]
        self.xDem, self.yDem, self.costDem = loadDEM(dem_df)
        self.nameWell, self.timeWell, self.x1Well, self.y1Well, self.z1Well, self.x3Well, self.y3Well, self.z3Well = loadWells(wells_df)
        self.well_names_control = copy(self.nameWell)
        self.xWellPad, self.yWellPad, self.wellNames = loadAnswer(answer_json)
        self.alpha = CONST_ALPHA
        self.c0 = CONST_C0
        self.c1 = CONST_C1
        self.c2 = CONST_C2
        self.c3 = CONST_C3
        self.const_l = CONST_L
        self.const_d = CONST_D
        self.penalty = CONST_PENALTY
        self.cost = 0
        self.status = True
        self.drill_timetable = []
        self.intersecting_wells = []
        self.log = ''
        self.well_pads = self.buildSolution(wells_df, dem_df)
        self.calculateCost()
        self.fig = 0
        
    def filter(self, pairs):
        result = []
        for pair in pairs:
            if (pair[0][1] != pair[1][1]):
                result.append((pair[0][0], pair[1][0]))
        return result
        
    def buildSolution(self, wells_df, dem_df):
        solution = []
        for well_pad_id, well_pad_wells in enumerate(self.wellNames):
            wp = WellPad(self.xWellPad[well_pad_id], self.yWellPad[well_pad_id], self.alpha, self.penalty)
            #add wells
            for well_name in self.wellNames[well_pad_id]:
                try:
                    index = self.nameWell.index(well_name)
                    #check that wells are in list
                except ValueError as e:
                    self.log += 'Value Error: invalid well name ' + str(well_name) + '\n'
                    self.cost += self.penalty
                    self.status = self.status and False
                    return solution
                self.well_names_control.remove(well_name)
                w = Well(wp, well_name, self.timeWell[index], self.x1Well[index], self.y1Well[index], self.z1Well[index], self.x3Well[index], self.y3Well[index], self.z3Well[index], self.c3, self.const_l, self.penalty)
                self.status = w.check_length() and self.status
                self.log += w.log
                self.drill_timetable.append((w, wp, well_pad_id))
            #check wp inside map
            self.status = wp.check_position(min(self.xDem), min(self.yDem), max(self.xDem), max(self.yDem)) and self.status
            self.status = wp.calculateK(self.xDem, self.yDem, self.costDem) and self.status
            self.log += 'wellPad K = ' + str(wp.k) + '\n'
            wp.update()
            self.log += wp.log
            solution.append(wp)
        if (len(self.well_names_control) > 0):
            self.status = self.status and False
            self.log += 'Error: wells ' + str(self.well_names_control) + ' are not drilled \n'
            self.cost += self.penalty * len(self.well_names_control) #penalty
            return solution
        #ckeck wells intersections
        w_pairs = itertools.combinations([(i[0], i[2]) for i in self.drill_timetable], 2)
        # filter - calculate distance only for wells from different well pads
        for pair in self.filter(w_pairs):
            d = self.calculateDistance(pair[0], pair[1])
            if (d < self.const_d):
                self.intersecting_wells.append((pair[0].name, pair[1].name, d))
            #print((pair[0].name, pair[1].name, d))
        if (len(self.intersecting_wells) > 0):
            self.log += 'Error: wells intersection ' + str(self.intersecting_wells) + '\n'
            self.cost += self.penalty * len(self.intersecting_wells) #penalty
            self.status = self.status and False
        self.drill_timetable.sort(key=lambda tup: tup[0].t)
        return solution
    
    def calculateCost(self):
        wp_id_curr = 0
        wp_id_prev = -1
        for index, well in enumerate(self.drill_timetable):
            wp_id_curr = well[2]
            if (wp_id_curr != wp_id_prev):
                if (wp_id_prev != -1):
                    prev_k = self.drill_timetable[index-1][1].k
                    self.cost += prev_k * self.c1
                    self.log += str(well[0].t) + ' - stop  drilling wells at Well Pad ' + str(wp_id_prev) + ', cost += ' + str(prev_k * self.c1) + ', cost = {:.2e} \n'.format(self.cost)    
                self.cost += well[1].k * self.c0
                self.log += str(well[0].t) + ' - start drilling wells at Well Pad ' + str(wp_id_curr) + ', cost += ' + str(well[1].k * self.c0) + ', cost = {:.2e} \n'.format(self.cost)   
            else:
                self.cost += well[1].k * self.c2
                self.log += str(well[0].t) + ' - move  drilling well ' + str(well[0].name) + ' at Well Pad ' + str(wp_id_curr) + ', cost += ' + str(well[1].k * self.c2) + ', cost = {:.2e} \n'.format(self.cost)
            self.cost += well[0].price
            self.log += str(well[0].t) + ' - done  drilling well ' + str(well[0].name) + ' at Well Pad ' + str(wp_id_curr) + ', cost += ' + str(well[0].price) + ', cost = {:.2e} \n'.format(self.cost)
            wp_id_prev = wp_id_curr
        
    def calculateDistance(self, well1, well2):
        d_min = 1e16
        for i1, coord1 in enumerate(well1.trajectory):
            for i2, coord2 in enumerate(well2.trajectory):
                #if (i1 > 0 and i2 > 0): #ignore init points if wells are from same well pad
                d = np.sqrt((coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2+(coord1[2]-coord2[2])**2)
                if (d < d_min):
                    d_min = d
        return d_min
    
    def get_grid_size(self, dataframe):
        X_len = [count for item, count in collections.Counter(dataframe["Y"]).items() if item == dataframe["Y"].min()]
        Y_len = [count for item, count in collections.Counter(dataframe["X"]).items() if item == dataframe["X"].min()]
        return X_len[0], Y_len[0]

    def get_axis(self, axis):
        return [item for item, count in collections.Counter(axis).items() if count > 1]
        
    def plot(self):
        fig, ax_scatter = plt.subplots(figsize=(120,120))
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1)
        #cmap = plt.cm.get_cmap('Wistia')
        cmap = plt.cm.get_cmap('jet')
        sc = plt.pcolormesh(self.X, self.Y, np.log(self.K), cmap=cmap)

        cb = plt.colorbar(sc)
        cb.ax.tick_params(labelsize=100)
        cb.ax.yaxis.offsetText.set_fontsize(100)
        #cb.ax.get_yaxis().labelpad = 50
        cb.ax.set_title('log(Koeff)', size=100)

        for wp in self.well_pads:
            #plot wells
            for well_name in wp.wells.keys():
                w = wp.wells[well_name]
                plt.plot(w.trajectory[:,0], w.trajectory[:,1], c='g', linewidth=10)
                #plt.annotate("{}: {:.2e}".format(w.name, w.price), (0.5*(w.x1+w.x3),0.5*(w.y1+w.y3)), size=30)
                #plt.scatter(w.xc, w.yc, c='orange', s=100)
                plt.scatter(w.x1, w.y1, c='k', s=700, marker="v", zorder=9992)
                plt.scatter(w.x3, w.y3, c='k', s=700, marker="^", zorder=9993)
            #plot well pads
            plt.scatter(wp.x, wp.y, c='r', s=400, label='Well Pad', zorder=9994)
            rect = patches.Rectangle((wp.x_min, wp.y_min), wp.a, wp.a, linewidth=1, edgecolor='grey', facecolor='grey')
            ax_scatter.add_patch(rect)

        #plot legend
        T1 = mlines.Line2D([], [], color="none", marker='v', markerfacecolor="k", markersize=50, label='Well T1')
        T3 = mlines.Line2D([], [], color="none", marker='^', markerfacecolor="k", markersize=50, label='Well T3')
        r = mlines.Line2D([], [], color="none", marker='s', markerfacecolor="grey", markersize=50, label='Well Pad')
        c = mlines.Line2D([], [], color="none", marker='o', markerfacecolor="r",  markersize=50, label='Well Pad center')
        l = mlines.Line2D([], [], color='green', marker='', lw=30,  markersize=30, label='Well trajectory')
        h = [T1, T3, r, c, l]
        plt.legend(handles=h, prop={'size': 100}, loc='best')

        #change text and ticks
        ax_scatter.set_xlabel('X', size=100)
        ax_scatter.set_ylabel('Y', size=100)
        ax_scatter.tick_params(axis='both', which='major', labelsize=100)
        ax_scatter.tick_params(axis='both', which='minor', labelsize=50)
        ax_scatter.xaxis.offsetText.set_fontsize(100)
        ax_scatter.yaxis.offsetText.set_fontsize(100)
        plt.title("TOTAL COST = {:.2e}".format(self.cost), size=100)
        plt.show() 
        self.fig = fig
    
    def save_answer(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.answer_json, fp,  indent=4)
            
    def save_plot(self, filename):
        #saving result
        self.fig.savefig(filename)
