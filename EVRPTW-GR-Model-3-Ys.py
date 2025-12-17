# -*- coding: utf-8 -*-
"""
EVRPTW with gradient- and load-dependent energy consumption + energy-based cuts.
EVRPTW-GR 
Model 3 - Minimising the SoC values in the Objective function


You only need to change:
    - num_v
    - filename (instance)

Author: Sina Rastani
"""
import numpy as np
import time
import warnings
from gurobipy import *
import pandas as pd
import gc
import os
import re

gc.enable()
warnings.filterwarnings("ignore")

# =============================================================================
#             READING DATA
# =============================================================================

def read_data_file(filename: str) -> pd.DataFrame:
    """Read whitespace-separated instance file into a pandas DataFrame."""
    try:
        data = pd.read_csv(filename, sep=r"\s+")
        return data
    except Exception:
        raise RuntimeError(f"Could not read file: {filename}")


def get_param_value(df: pd.DataFrame, key: str, default: float | None = None) -> float:
    # 1) Try Schneider/Desaulniers style: StringID == key
    if "StringID" in df.columns:
        row = df.loc[df["StringID"] == key]
        if not row.empty:
            row = row.iloc[0]
            for val in row.values:
                if isinstance(val, str) and "/" in val:
                    for token in val.split("/"):
                        token = token.strip()
                        if token:
                            try:
                                return float(token)
                            except ValueError:
                                continue

    # 2) More general scan over rows mentioning the key
    for _, row in df.iterrows():
        if "StringID" in df.columns:
            sid = row["StringID"]
            if isinstance(sid, str) and key not in sid:
                continue
        for val in row.values:
            if isinstance(val, str) and "/" in val:
                for token in val.split("/"):
                    token = token.strip()
                    if token:
                        try:
                            return float(token)
                        except ValueError:
                            continue

    if default is not None:
        return default

    raise RuntimeError(
        f"Parameter '{key}' not found in data file and no default was provided."
    )


def build_instance_path(filename, base_dir="Data"):
    """
    Build full path to data file based on:
      - number of customers:
          * Old style:  C5, C10, C15, C25, C100  (e.g. c101C5_L.txt)
          * New style:  50_21  (e.g. c101_50_21_L.txt)
      - slope type: _L, _NL, _VG

    Folder structure:
      Data/<NetworkGroup>/<X_Customers>/<SlopeFolder>/<filename>

    where:
      Small_Network : C5, C10, C15
      Medium_Network: C25
      Large_Network : C50, C100
    """
    # extract customer size
    m = re.search(r'C(\d+)', filename)
    if m:
        n_cust = int(m.group(1))
    else:
        m2 = re.search(r'_(\d+)_\d+_', filename)  # e.g. _50_21_
        if not m2:
            raise ValueError(f"Cannot infer customer size from filename: {filename}")
        n_cust = int(m2.group(1))

    # map to network group
    if n_cust in (5, 10, 15):
        network_group = "Small_Network"
    elif n_cust == 25:
        network_group = "Medium_Network"
    elif n_cust in (50, 100):
        network_group = "Large_Network"
    else:
        raise ValueError(f"Unsupported customer size: {n_cust} in filename: {filename}")

    size_folder = f"{n_cust}_Customers"

    # slope folder
    if filename.endswith("_L.txt"):
        slope_folder = "Level"
    elif filename.endswith("_NL.txt"):
        slope_folder = "Nearly_Level"
    elif filename.endswith("_VG.txt"):
        slope_folder = "Very_Gentle"
    else:
        raise ValueError(f"Unknown slope suffix in filename: {filename}")

    full_path = os.path.join(base_dir,
                             network_group,
                             size_folder,
                             slope_folder,
                             filename)
    return full_path


# =============================================================================
# BASIC SETTINGS & INSTANCE SELECTION
# =============================================================================

num_v = 2  # number of electric vehicles

# -------------------------------------------------------------------------
# Instance selection (activate exactly one)
# -------------------------------------------------------------------------

# --- 5 customers --------------------------------------------------------
# filename = "r104C5_L.txt"
filename = "r104C5_NL.txt"
# filename = "r104C5_VG.txt"

# filename = "r105C5_L.txt"
# filename = "r105C5_NL.txt"
# filename = "r105C5_VG.txt"

# filename = "r202C5_L.txt"
# filename = "r202C5_NL.txt"
# filename = "r202C5_VG.txt"

# filename = "r203C5_L.txt"
# filename = "r203C5_NL.txt"
# filename = "r203C5_VG.txt"

# filename = "c101C5_L.txt"
# filename = "c101C5_NL.txt"
# filename = "c101C5_VG.txt"

# filename = "c103C5_L.txt"
# filename = "c103C5_NL.txt"
# filename = "c103C5_VG.txt"

# filename = "c206C5_L.txt"
# filename = "c206C5_NL.txt"
# filename = "c206C5_VG.txt"

# filename = "c208C5_L.txt"
# filename = "c208C5_NL.txt"
# filename = "c208C5_VG.txt"

# filename = "rc105C5_L.txt"
# filename = "rc105C5_NL.txt"
# filename = "rc105C5_VG.txt"

# filename = "rc108C5_L.txt"
# filename = "rc108C5_NL.txt"
# filename = "rc108C5_VG.txt"

# filename = "rc204C5_L.txt"
# filename = "rc204C5_NL.txt"
# filename = "rc204C5_VG.txt"

# filename = "rc208C5_L.txt"
# filename = "rc208C5_NL.txt"
# filename = "rc208C5_VG.txt"

# --- 10 customers -------------------------------------------------------
# filename = "r102C10_L.txt"
# filename = "r102C10_NL.txt"
# filename = "r102C10_VG.txt"

# filename = "r103C10_L.txt"
# filename = "r103C10_NL.txt"
# filename = "r103C10_VG.txt"

# filename = "r201C10_L.txt"
# filename = "r201C10_NL.txt"
# filename = "r201C10_VG.txt"

# filename = "r203C10_L.txt"
# filename = "r203C10_NL.txt"
# filename = "r203C10_VG.txt"

# filename = "c101C10_L.txt"
# filename = "c101C10_NL.txt"
# filename = "c101C10_VG.txt"

# filename = "c104C10_L.txt"
# filename = "c104C10_NL.txt"
# filename = "c104C10_VG.txt"

# filename = "c202C10_L.txt"
# filename = "c202C10_NL.txt"
# filename = "c202C10_VG.txt"

# filename = "c205C10_L.txt"
# filename = "c205C10_NL.txt"
# filename = "c205C10_VG.txt"

# filename = "rc102C10_L.txt"
# filename = "rc102C10_NL.txt"
# filename = "rc102C10_VG.txt"

# filename = "rc108C10_L.txt"
# filename = "rc108C10_NL.txt"
# filename = "rc108C10_VG.txt"

# filename = "rc201C10_L.txt"
# filename = "rc201C10_NL.txt"
# filename = "rc201C10_VG.txt"

# filename = "rc205C10_L.txt"
# filename = "rc205C10_NL.txt"
# filename = "rc205C10_VG.txt"

# --- 15 customers -------------------------------------------------------
# filename = "r102C15_L.txt"
# filename = "r102C15_NL.txt"
# filename = "r102C15_VG.txt"

# filename = "r105C15_L.txt"
# filename = "r105C15_NL.txt"
# filename = "r105C15_VG.txt"

# filename = "r202C15_L.txt"
# filename = "r202C15_NL.txt"
# filename = "r202C15_VG.txt"

# filename = "r209C15_L.txt"
# filename = "r209C15_NL.txt"
# filename = "r209C15_VG.txt"

# filename = "c103C15_L.txt"
# filename = "c103C15_NL.txt"
# filename = "c103C15_VG.txt"

# filename = "c106C15_L.txt"
# filename = "c106C15_NL.txt"
# filename = "c106C15_VG.txt"

# filename = "c202C15_L.txt"
# filename = "c202C15_NL.txt"
# filename = "c202C15_VG.txt"

# filename = "c208C15_L.txt"
# filename = "c208C15_NL.txt"
# filename = "c208C15_VG.txt"

# filename = "rc103C15_L.txt"
# filename = "rc103C15_NL.txt"
# filename = "rc103C15_VG.txt"

# filename = "rc108C15_L.txt"
# filename = "rc108C15_NL.txt"
# filename = "rc108C15_VG.txt"

# filename = "rc202C15_L.txt"
# filename = "rc202C15_NL.txt"
# filename = "rc202C15_VG.txt"

# filename = "rc204C15_L.txt"
# filename = "rc204C15_NL.txt"
# filename = "rc204C15_VG.txt"

# --- Medium_Network / 25_Customers -------------------------------------

# filename = "c101C25_L.txt"
# filename = "c101C25_NL.txt"
# filename = "c101C25_VG.txt"

# filename = "c102C25_L.txt"
# filename = "c102C25_NL.txt"
# filename = "c102C25_VG.txt"

# filename = "c104C25_L.txt"
# filename = "c104C25_NL.txt"
# filename = "c104C25_VG.txt"

# filename = "c108C25_L.txt"
# filename = "c108C25_NL.txt"
# filename = "c108C25_VG.txt"

# filename = "r102C25_L.txt"
# filename = "r102C25_NL.txt"
# filename = "r102C25_VG.txt"

# filename = "r104C25_L.txt"
# filename = "r104C25_NL.txt"
# filename = "r104C25_VG.txt"

# filename = "r107C25_L.txt"
# filename = "r107C25_NL.txt"
# filename = "r107C25_VG.txt"

# filename = "r110C25_L.txt"
# filename = "r110C25_NL.txt"
# filename = "r110C25_VG.txt"

# filename = "rc101C25_L.txt"
# filename = "rc101C25_NL.txt"
# filename = "rc101C25_VG.txt"

# filename = "rc103C25_L.txt"
# filename = "rc103C25_NL.txt"
# filename = "rc103C25_VG.txt"

# filename = "rc104C25_L.txt"
# filename = "rc104C25_NL.txt"
# filename = "rc104C25_VG.txt"

# filename = "rc106C25_L.txt"
# filename = "rc106C25_NL.txt"
# filename = "rc106C25_VG.txt"

# --- Large_Network / 50_Customers -------------------------------------

# filename = "c101_50_21_L.txt"
# filename = "c101_50_21_NL.txt"
# filename = "c101_50_21_VG.txt"

# filename = "c102_50_21_L.txt"
# filename = "c102_50_21_NL.txt"
# filename = "c102_50_21_VG.txt"

# filename = "r102_50_21_L.txt"
# filename = "r102_50_21_NL.txt"
# filename = "r102_50_21_VG.txt"

# filename = "r107_50_21_L.txt"
# filename = "r107_50_21_NL.txt"
# filename = "r107_50_21_VG.txt"


# --- Large_Network / 100_Customers -------------------------------------

# filename = "c101C100_L.txt"
# filename = "c101C100_NL.txt"
# filename = "c101C100_VG.txt"

# filename = "c102C100_L.txt"
# filename = "c102C100_NL.txt"
# filename = "c102C100_VG.txt"

# filename = "c104C100_L.txt"
# filename = "c104C100_NL.txt"
# filename = "c104C100_VG.txt"

# filename = "c108C100_L.txt"
# filename = "c108C100_NL.txt"
# filename = "c108C100_VG.txt"

# filename = "r102C100_L.txt"
# filename = "r102C100_NL.txt"
# filename = "r102C100_VG.txt"

# filename = "r104C100_L.txt"
# filename = "r104C100_NL.txt"
# filename = "r104C100_VG.txt"

# filename = "r107C100_L.txt"
# filename = "r107C100_NL.txt"
# filename = "r107C100_VG.txt"

# filename = "r110C100_L.txt"
# filename = "r110C100_NL.txt"
# filename = "r110C100_VG.txt"

# filename = "rc101C100_L.txt"
# filename = "rc101C100_NL.txt"
# filename = "rc101C100_VG.txt"

# filename = "rc103C100_L.txt"
# filename = "rc103C100_NL.txt"
# filename = "rc103C100_VG.txt"

# filename = "rc104C100_L.txt"
# filename = "rc104C100_NL.txt"
# filename = "rc104C100_VG.txt"

# filename = "rc106C100_L.txt"
# filename = "rc106C100_NL.txt"
# filename = "rc106C100_VG.txt"


grad_mult     = 1.0       # scaling for altitude
h             = 1.0
truncation    = 0         # 0 = Euclidean, 1 = truncated * 100
write_in_text = 0         # 1 to write to text file

hhh = str(h).replace(".", ",")

# =============================================================================
# READ INSTANCE DATA (ALTITUDE IN SAME FILE)
# =============================================================================

filepath = build_instance_path(filename)
data = read_data_file(filepath)
if data is None:
    raise RuntimeError(f"Data file could not be loaded: {filepath}")

# separate nodes (d, f, c) from parameter rows
nodes = data[data["Type"].isin(["d", "f", "c"])].reset_index(drop=True)

ns = int(sum(nodes["Type"] == "f"))  # number of stations
nc = int(sum(nodes["Type"] == "c"))  # number of customers

# parameters from bottom lines of data file
Q               = get_param_value(data, "Q")      # battery capacity
raw_vehicle_cap = get_param_value(data, "C")      # original load capacity
g               = get_param_value(data, "g")      # inverse refuelling rate
v_speed         = get_param_value(data, "v")      # average velocity (should be 1.0)

# we work in 3650-capacity space (demands already scaled in data)
vehicle_cap = 3650.0

# node order: depot (0), customers (1..nc), stations (nc+1..nc+ns), depot-end (nc+ns+1)
    
    
    
early_arrive1 = list(data.loc[0:0,'ReadyTime'].astype(float)) + \
                            list(data.loc[1 + ns:nc + ns,'ReadyTime'].astype(float)) +\
                            list(data.loc[1 : ns,'ReadyTime'].astype(float))  + \
                            list(data.loc[0:0,'ReadyTime'].astype(float))
late_arrive1 = list(data.loc[0:0,'DueDate'].astype(float)) + \
                            list(data.loc[1 + ns:nc + ns,'DueDate'].astype(float)) +\
                            list(data.loc[1 : ns,'DueDate'].astype(float))  + \
                            list(data.loc[0:0,'DueDate'].astype(float))
                            
service_time1 = list(data.loc[0:0,'ServiceTime'].astype(float)) + \
                            list(data.loc[1 + ns:nc + ns,'ServiceTime'].astype(float)) +\
                            list(data.loc[1 : ns,'ServiceTime'].astype(float))  + \
                            list(data.loc[0:0,'ServiceTime'].astype(float))
                            
demand1 = list(data.loc[0:0,'demand'].astype(float)) + \
                            list(data.loc[1 + ns:nc + ns,'demand'].astype(float)) +\
                            list(data.loc[1 : ns,'demand'].astype(float))  + \
                            list(data.loc[0:0,'demand'].astype(float))      

altitude1 = list(data.loc[0:0,'altitude'].astype(float)) + \
                            list(data.loc[1 + ns:nc + ns,'altitude'].astype(float)) +\
                            list(data.loc[1 : ns,'altitude'].astype(float))  + \
                            list(data.loc[0:0,'altitude'].astype(float))    
                            

                        
                            
depot_x = data[data['Type'] == 'd']['x'].astype(float)[0]
depot_y = data[data['Type'] == 'd']['y'].astype(float)[0]                            
total_node = nc + ns + 2*num_v                        
distance = np.zeros((total_node, total_node))

cordinate_x1 =   list(data.loc[0:0,'x'].astype(float)) + \
                list(data.loc[1 + ns:nc + ns,'x'].astype(float)) + \
                list(data.loc[1 : ns,'x'].astype(float)) + \
                list(data.loc[0:0,'x'].astype(float))
                            

cordinate_y1 =   list(data.loc[0:0,'y'].astype(float)) + \
                list(data.loc[1 + ns:nc + ns,'y'].astype(float)) + \
                list(data.loc[1 : ns,'y'].astype(float)) + \
                list(data.loc[0:0,'y'].astype(float))
      
        
early_arrive = []
late_arrive = []
service_time = []
demand = []
cordinate_x = []
cordinate_y = []

for i in range(num_v - 1):
    early_arrive.append(float(data.loc[0, 'ReadyTime']))
    late_arrive.append(float(data.loc[0, 'DueDate']))
    service_time.append(float(data.loc[0, 'ServiceTime']))
    demand.append(float(data.loc[0, 'demand']))
    cordinate_x.append(float(data.loc[0, 'x']))
    cordinate_y.append(float(data.loc[0, 'y']))

for i in range(len(early_arrive1)):
    early_arrive.append(early_arrive1[i])
    late_arrive.append(late_arrive1[i])
    service_time.append(service_time1[i])
    demand.append(demand1[i])
    cordinate_x.append(cordinate_x1[i])
    cordinate_y.append(cordinate_y1[i])

for i in range(num_v - 1):
    early_arrive.append(float(data.loc[0, 'ReadyTime']))
    late_arrive.append(float(data.loc[0, 'DueDate']))
    service_time.append(float(data.loc[0, 'ServiceTime']))
    demand.append(float(data.loc[0, 'demand']))
    cordinate_x.append(float(data.loc[0, 'x']))
    cordinate_y.append(float(data.loc[0, 'y']))
    

# depot altitude as a float
depot_alt = float(data.loc[0, 'altitude'])

# customers altitude block
cust_alt = list(data.loc[1 + ns:nc + ns, 'altitude'].astype(float))

# station altitude block
stat_alt = list(data.loc[1:ns, 'altitude'].astype(float))

# construct full altitude list:
# dummy depots + real depot + customers + stations + real depot + dummy depots
altitude = (
    [depot_alt] * (num_v-1) +       # dummy departure depots
    [depot_alt] +               # first depot
    cust_alt +                  # customers
    stat_alt +                  # stations
    [depot_alt] +               # final depot
    [depot_alt] * (num_v-1)         # dummy arrival depots
)

cutss = 1                 # 1 = use cuts, 0 = no cuts
relax = 0                 # 0 = binary x,z ; 1 = LP relaxation
h = 1.0
truncation = 0
v_speed = 1.0
write_in_text = 0         # 1 = write to file, 0 = console only


vehicle_cap = 3650

# distances and travel times
if truncation == 1:
    for i in range(total_node):
        for j in range(total_node):
            dist_temp = np.sqrt((cordinate_x[i] - cordinate_x[j]) ** 2 +
                                (cordinate_y[i] - cordinate_y[j]) ** 2)
            if 100 * dist_temp - int(100 * dist_temp) < 1e-9:
                distance[i, j] = 100 * dist_temp
            else:
                distance[i, j] = int(100 * dist_temp) + 1
    travel_time = distance / 100 * v_speed
    Q = 100 * Q
    g = g / 100
else:
    for i in range(total_node):
        for j in range(total_node):
            distance[i, j] = np.sqrt((cordinate_x[i] - cordinate_x[j]) ** 2 +
                                     (cordinate_y[i] - cordinate_y[j]) ** 2)
    travel_time = distance / v_speed


# =============================================================================
#                             RANGES AND EXPRESSIONS
# =============================================================================

late_arrive0 = late_arrive[0]

customers = range(num_v, num_v + nc)               # V (customers)
stations = range(num_v + nc, num_v + nc + ns)      # F (stations)
v_total = range(0, total_node)

cust = tuple(customers)

customers_prime = ()
v0 = ()
dumm_depot_arrival = ()
dumm_depot_departure = ()

for i in range(num_v + nc + ns, total_node):
    dumm_depot_arrival += (i,)
    dumm_depot_departure += (i - num_v - nc - ns,)

v0 = dumm_depot_departure + cust
customers_prime = cust + dumm_depot_arrival
vpr0 = v0 + dumm_depot_arrival

thatijs = np.zeros((total_node, total_node, ns + num_v))
for i in range(total_node):
    for j in range(total_node):
        for s in stations:
            thatijs[i, j, s - nc - 1] = travel_time[i, s] + travel_time[s, j] - travel_time[i, j]

# extended node sets for energy variables
stat_tuple = tuple(stations)
v00 = v0 + stat_tuple
vpr00 = cust + stat_tuple + dumm_depot_arrival

# =============================================================================
#                       PHYSICAL ENERGY MODEL (ACCURATE)
# =============================================================================

weight = 6350.0
load = 0.0
vv = 60.0
v_s = vv / 3.6           # 60 km/h in m/s
gg = 9.81
cr = 0.01                # rolling resistance
cd = 0.7                 # aerodynamic drag
ro = 1.2041              # air density
a = 3.912                # frontal area
kk = 1.0
conv_f = 1.0
KNV = 0.0                # EV, no fuel term
p_acc = 0.0
degree = 0.0
acceleration = 0.0

# realistic engine/train efficiencies
mu = 0.9
mu_train = 0.9
regenerating = 0.8


angle = np.zeros((len(distance), len(distance)))
for i in range(len(distance)):
    for j in range(len(distance)):
        if distance[i, j] == 0:
            angle[i, j] = 0
        else:
            angle[i, j] = np.degrees(np.arctan((altitude[j] - altitude[i]) / distance[i, j]))

print('Average slope: ', np.average(np.abs(angle)))

# base consumption on flat, empty
p_tract_base = ((weight + load) * acceleration +
                (weight + load) * gg * np.sin(np.deg2rad(degree)) +
                0.5 * cd * ro * a * v_s * v_s +
                (weight + load) * gg * cr * np.cos(np.deg2rad(degree))) * v_s / 1000.0
p_total_base = (KNV + (p_tract_base / mu_train + p_acc) / mu) / (kk * conv_f)
consump_base = p_total_base / vv

# full matrix of consumption for load=0
p_tract = ((weight + load) * acceleration +
           (weight + load) * gg * np.sin(np.deg2rad(angle)) +
           0.5 * cd * ro * a * v_s * v_s +
           (weight + load) * gg * cr * np.cos(np.deg2rad(angle))) * v_s / 1000.0
p_total = np.zeros_like(p_tract)

for i in range(len(p_tract)):
    for j in range(len(p_tract)):
        if p_tract[i, j] > 0:
            p_total[i, j] = (KNV + (p_tract[i, j] / mu_train + p_acc) / mu) / (kk * conv_f)
        else:
            p_total[i, j] = regenerating * (KNV + (p_tract[i, j] + p_acc)) / (kk * conv_f)

consump = p_total / vv

extra_load = np.sum(demand)
p_tract_max = ((weight + extra_load) * acceleration +
               (weight + extra_load) * gg * np.sin(np.deg2rad(np.max(angle))) +
               0.5 * cd * ro * a * v_s * v_s +
               (weight + extra_load) * gg * cr * np.cos(np.deg2rad(np.max(angle)))) * v_s / 1000.0
p_total_max = (KNV + (p_tract_max / mu_train + p_acc) / mu) / (kk * conv_f)
consump_max = p_total_max / vv
mx_consump = consump_max / consump_base

# =============================================================================
#                 ENERGY BOUNDS FOR CUT GENERATION (min/max)
# =============================================================================

def min_energy_consumption(s, i):
    """Minimum normalized energy-per-distance over load in [0, sum(demand)]."""
    energy_consump_list = []
    for load_val in [0, sum(demand)]:
        p_tract_loc = ((weight + load_val) * acceleration +
                       (weight + load_val) * gg * np.sin(np.deg2rad(angle[s, i])) +
                       0.5 * cd * ro * a * v_s * v_s +
                       (weight + load_val) * gg * cr * np.cos(np.deg2rad(angle[s, i]))) * v_s / 1000.0
        if p_tract_loc >= 0:
            p_total_loc = (KNV + (p_tract_loc / mu_train + p_acc) / mu) / (kk * conv_f)
            consump_loc = p_total_loc / vv
            energy_consump = consump_loc / consump_base
        else:
            p_total_loc = regenerating * (KNV + (p_tract_loc + p_acc)) / (kk * conv_f)
            consump_loc = p_total_loc / vv
            energy_consump = consump_loc / consump_base
        energy_consump_list.append(energy_consump)
    return min(energy_consump_list)


def max_energy_consumption(s, i):
    """Maximum normalized energy-per-distance over load in [0, sum(demand)]."""
    energy_consump_list = []
    for load_val in [0, sum(demand)]:
        p_tract_loc = ((weight + load_val) * acceleration +
                       (weight + load_val) * gg * np.sin(np.deg2rad(angle[s, i])) +
                       0.5 * cd * ro * a * v_s * v_s +
                       (weight + load_val) * gg * cr * np.cos(np.deg2rad(angle[s, i]))) * v_s / 1000.0
        if p_tract_loc >= 0:
            p_total_loc = (KNV + (p_tract_loc / mu_train + p_acc) / mu) / (kk * conv_f)
            consump_loc = p_total_loc / vv
            energy_consump = consump_loc / consump_base
        else:
            p_total_loc = regenerating * (KNV + (p_tract_loc + p_acc)) / (kk * conv_f)
            consump_loc = p_total_loc / vv
            energy_consump = consump_loc / consump_base
        energy_consump_list.append(energy_consump)
    return max(energy_consump_list)

# =============================================================================
#                       CONNECTIVITY CUTS (FROM OLD MILP3)
# =============================================================================

def connectivity_cuts(v_dummy):
    i_stat = []
    j_stat = []
    k_stat = []

    stat_dominators = np.ones(
        (len(dumm_depot_arrival) + len(dumm_depot_departure) + nc + ns,
         len(dumm_depot_arrival) + len(dumm_depot_departure) + nc + ns,
         ns + 1)
    )

    dominant_stations_dic = {}

    max_index = len(dumm_depot_arrival) + len(dumm_depot_departure) + ns + nc
    for idx, i in enumerate(range(max_index)):
        if len(dumm_depot_departure) + nc - 1 < i < len(dumm_depot_departure) + nc + ns - 1:
            continue
        for jdx, j in enumerate(range(max_index)):
            if len(dumm_depot_departure) + nc - 1 < j < len(dumm_depot_departure) + ns + nc - 1:
                continue
            if i == j:
                continue
            if ((i in dumm_depot_departure and j in dumm_depot_arrival) or
                (i in dumm_depot_arrival and j in dumm_depot_departure)):
                continue

            dominant = list(stations)
            for k in range(len(dumm_depot_departure) + nc - 1,
                           len(dumm_depot_departure) + nc + ns - 1):
                for s in list(dominant):
                    if k != s:
                        if (distance[i, k] * max_energy_consumption(i, k) <=
                            distance[i, s] * min_energy_consumption(i, s) and
                            distance[k, j] * max_energy_consumption(k, j) <=
                            distance[s, j] * min_energy_consumption(s, j)):
                            dominant.remove(s)
            dominant_stations_dic[i, j] = dominant

    # max_soc and least_soc, earliest_possible
    max_soc = []
    for i in range(max_index):
        possible_dep_soc = []
        for s in stations:
            possible_dep_soc.append(Q - distance[s, i] * min_energy_consumption(s, i))
        max_soc.append(max(possible_dep_soc))

    least_soc = []
    for i in range(max_index):
        possible_arr_soc = []
        for s in stations:
            possible_arr_soc.append(distance[s, i] * min_energy_consumption(i, s))
        least_soc.append(max(0, min(possible_arr_soc)))

    earliest_possible = []
    for i in range(max_index):
        earliest_possible.append(max(early_arrive[i], travel_time[0, i]))

    connect_cuts = np.zeros_like(distance)
    cust_sets = []
    new_cut_sets = {}

    for i in range(max_index):
        if len(dumm_depot_departure) + nc - 1 < i <= len(dumm_depot_departure) + nc + ns - 1:
            continue
        for j in range(max_index):
            if (len(dumm_depot_departure) + nc - 1 < j <= len(dumm_depot_departure) + nc + ns - 1 or
                i == j):
                continue

            if (demand[i] + demand[j] > vehicle_cap or
                earliest_possible[i] > late_arrive[i] or
                earliest_possible[i] + service_time[i] + travel_time[i, j] > late_arrive[j] or
                max(earliest_possible[i] + service_time[i] + travel_time[i, j], early_arrive[j]) +
                service_time[j] + travel_time[j, max_index - 1] > late_arrive[max_index - 1]):
                continue
            else:
                if max_soc[i] - min_energy_consumption(i, j) * distance[i, j] - least_soc[j] >= 0:
                    connect_cuts[i, j] = 1
                    if (j not in dumm_depot_departure) and (i not in dumm_depot_arrival):
                        cust_sets.append((i, j))
                else:
                    for k in dominant_stations_dic[i, j]:
                        if (max_soc[i] - min_energy_consumption(i, k) * distance[i, k] >= 0 and
                            min_energy_consumption(k, j) * distance[k, j] + least_soc[j] <= Q):
                            recharge = (min_energy_consumption(k, j) * distance[k, j] +
                                        least_soc[j] -
                                        (max_soc[i] - min_energy_consumption(i, k) * distance[i, k]))
                            if (recharge <= Q and
                                (max_soc[i] - min_energy_consumption(i, k) * distance[i, k]) +
                                recharge <= Q):
                                if (earliest_possible[i] + service_time[i] + travel_time[i, k] <=
                                    late_arrive[k] and
                                    max(earliest_possible[i] + service_time[i] +
                                        travel_time[i, k], early_arrive[k]) +
                                    recharge * g + travel_time[k, j] <= late_arrive[j] and
                                    max(max(earliest_possible[i] + service_time[i] +
                                            travel_time[i, k], early_arrive[k]) +
                                        recharge * g + travel_time[k, j], early_arrive[j]) +
                                    travel_time[j, max_index - 1] <= late_arrive[max_index - 1]):
                                    connect_cuts[i, j] = 1
                                    if (j not in dumm_depot_departure) and (i not in dumm_depot_arrival):
                                        cust_sets.append((i, j))
                                    if i not in dumm_depot_arrival:
                                        new_cut_sets[i, j] = dominant_stations_dic[i, j]
                                    break

    return connect_cuts, stat_dominators, new_cut_sets, cust_sets

# Run cut generation
(connect_cuts, stat_dominators, new_cut_sets, cust_sets) = connectivity_cuts(v_speed)
stat_dominators[:, :, 0] = 0  # as in original

customer_cuts_set = list(new_cut_sets.keys())
for idx, (i, j) in enumerate(customer_cuts_set):
    # if arc goes to dummy depot departure, map to final dummy arrival index
    if j in dumm_depot_departure:
        customer_cuts_set[idx] = [i, len(dumm_depot_arrival) + len(dumm_depot_departure) + nc + ns - 1]

LB_Veh = np.ceil(np.sum(demand) / vehicle_cap)

# =============================================================================
#                             OPTIMIZATION MODEL
# =============================================================================

start_time = time.time()
evrp = Model('EVRP')

# Decision variables
t = evrp.addVars(v_total, lb=0.0, vtype=GRB.CONTINUOUS, name='t')

if relax == 1:
    z = evrp.addVars(v0, customers_prime, stations, lb=0.0, vtype=GRB.CONTINUOUS, name='z')
    x = evrp.addVars(v0, customers_prime, lb=0.0, vtype=GRB.CONTINUOUS, name='x')
else:
    z = evrp.addVars(v0, customers_prime, stations, lb=0.0, vtype=GRB.BINARY, name='z')
    x = evrp.addVars(v0, customers_prime, lb=0.0, vtype=GRB.BINARY, name='x')

y = evrp.addVars(vpr0, lb=0.0, vtype=GRB.CONTINUOUS, name='y')
YYijs = evrp.addVars(v0, customers_prime, stations, lb=0.0, vtype=GRB.CONTINUOUS, name='YYijs')
yijs = evrp.addVars(v0, customers_prime, stations, lb=0.0, vtype=GRB.CONTINUOUS, name='yijs')

hh = evrp.addVars(v00, vpr00, lb=-Q, vtype=GRB.CONTINUOUS, name='hh')
zz = evrp.addVars(v00, vpr00, lb=0.0, vtype=GRB.BINARY, name='zz')
ptractive = evrp.addVars(v00, vpr00, lb=-Q, vtype=GRB.CONTINUOUS, name='ptractive')

u = evrp.addVars(v00, vpr00, lb=0.0, vtype=GRB.CONTINUOUS, name='u')
uu = evrp.addVars(v00, vpr00, stations, lb=0.0, vtype=GRB.CONTINUOUS, name='uu')

# ---------------- Capacity & load propagation ----------------

# vehicle leaves depot with empty load + weight
evrp.addConstrs((u[i, j] == weight * x[i, j])
                for i in dumm_depot_departure for j in customers_prime)

# load bounds on customer arcs
evrp.addConstrs((demand[i] * x[i, j] + weight * x[i, j] <= u[i, j])
                for i in customers for j in customers_prime if i != j)

evrp.addConstrs((u[i, j] <= (weight + np.sum(demand)) * x[i, j])
                for i in customers for j in customers_prime)

# propagate load between arcs
evrp.addConstrs(
    (u[j, k] >= quicksum(u[i, j] for i in v0) + demand[j] * x[j, k] -
     (weight + vehicle_cap) * (1 - x[j, k]))
    for j in customers for k in customers_prime
)

evrp.addConstrs(
    (u[j, k] <= quicksum(u[i, j] for i in v0) + demand[j] * x[j, k] +
     (weight + vehicle_cap) * (1 - x[j, k]))
    for j in customers for k in customers_prime
)

# load on arcs via station (uu)
evrp.addConstrs((uu[i, j, s] == weight * z[i, j, s])
                for i in dumm_depot_departure for j in customers_prime for s in stations)

evrp.addConstrs(
    (uu[j, k, s] >= quicksum(u[i, j] for i in v0) + demand[j] * z[j, k, s] -
     (weight + vehicle_cap) * (1 - z[j, k, s]))
    for j in customers for k in customers_prime for s in stations
)

evrp.addConstrs(
    (uu[j, k, s] <= quicksum(u[i, j] for i in v0) + demand[j] * z[j, k, s] +
     (weight + vehicle_cap) * (1 - z[j, k, s]))
    for j in customers for k in customers_prime for s in stations
)

evrp.addConstrs(
    (uu[i, j, s] <= z[i, j, s] * (np.sum(demand) + weight)
     for i in customers for j in customers_prime for s in stations)
)

# ---------------- Routing structure ----------------

evrp.addConstrs(
    (quicksum(x[i, j] for j in customers_prime if i != j) == 1)
    for i in customers
)

evrp.addConstrs(
    (quicksum(x[i, j] for i in v0 if i != j) -
     quicksum(x[j, i] for i in customers_prime if i != j) == 0)
    for j in customers
)

evrp.addConstrs(
    (quicksum(z[i, j, s] for s in stations) <= x[i, j])
    for i in v0 for j in customers_prime
)

# time windows
evrp.addConstrs(
    (t[i] + x[i, j] * (travel_time[i, j] + service_time[i]) +
     quicksum(z[i, j, s] * thatijs[i, j, s - nc - 1] +
              g * (YYijs[i, j, s] - yijs[i, j, s])
              for s in stations) -
     late_arrive0 * (1 - x[i, j]) <= t[j])
    for i in v0 for j in customers_prime if i != j
)

evrp.addConstrs((t[j] >= early_arrive[j]) for j in v_total)
evrp.addConstrs((t[j] <= late_arrive[j]) for j in v_total)

# ---------------- Consumption calculations (ptractive, hh) ----------------

# arcs without explicit station in between (direct part)
evrp.addConstrs(
    (ptractive[i, j] >=
     ((KNV + (((u[i, j] - quicksum(uu[i, j, s] for s in stations)) * acceleration +
               (u[i, j] - quicksum(uu[i, j, s] for s in stations)) *
               gg * np.sin(np.deg2rad(angle[i, j])) +
               (x[i, j] - quicksum(z[i, j, s] for s in stations)) *
               0.5 * cd * ro * a * v_s * v_s +
               (u[i, j] - quicksum(uu[i, j, s] for s in stations)) *
               gg * cr * np.cos(np.deg2rad(angle[i, j]))) * v_s / 1000.0)) /
      (kk * conv_f)) / vv)
     for i in v0 for j in customers_prime)


# consumption on station-to-customer arcs
evrp.addConstrs(
    (ptractive[s, j] >=
     ((KNV + ((quicksum(uu[i, j, s] for i in v0) * acceleration +
               quicksum(uu[i, j, s] for i in v0) *
               gg * np.sin(np.deg2rad(angle[s, j])) +
               quicksum(z[i, j, s] for i in v0) *
               0.5 * cd * ro * a * v_s * v_s +
               quicksum(uu[i, j, s] for i in v0) *
               gg * cr * np.cos(np.deg2rad(angle[s, j]))) * v_s / 1000.0)) /
      (kk * conv_f)) / vv)
     for s in stations for j in customers_prime)


# consumption on customer-to-station arcs
evrp.addConstrs(
    (ptractive[i, s] >=
     ((KNV + ((quicksum(uu[i, j, s] for j in customers_prime) * acceleration +
               quicksum(uu[i, j, s] for j in customers_prime) *
               gg * np.sin(np.deg2rad(angle[i, s])) +
               quicksum(z[i, j, s] for j in customers_prime) *
               0.5 * cd * ro * a * v_s * v_s +
               quicksum(uu[i, j, s] for j in customers_prime) *
               gg * cr * np.cos(np.deg2rad(angle[i, s]))) * v_s / 1000.0)) /
      (kk * conv_f)) / vv)
     for i in v0 for s in stations)


# link ptractive and hh via zz
evrp.addConstrs(
    (Q * zz[i, j] >= ptractive[i, j])
    for i in v00 for j in vpr00 if i != j
)

evrp.addConstrs(
    (Q * (zz[i, j] - 1) <= ptractive[i, j])
    for i in v00 for j in vpr00 if i != j
)

evrp.addConstrs(
    (hh[i, j] >= Q * (zz[i, j] - 1) + (ptractive[i, j] / (mu_train * mu)) / consump_base)
    for i in v00 for j in vpr00 if i != j
)

evrp.addConstrs(
    (hh[i, j] >= -Q * zz[i, j] + regenerating * ptractive[i, j] / consump_base)
    for i in v00 for j in vpr00 if i != j
)

# no self-consumption on station-to-station (optional but you had it)
evrp.addConstrs((hh[i, j] == 0) for i in stations for j in stations)

# ---------------- SOC (y) constraints with hh ----------------

evrp.addConstrs(
    (y[j] <= y[i] - hh[i, j] * distance[i, j] +
     (Q + mx_consump * np.max(distance)) *
     (1 - x[i, j] + quicksum(z[i, j, s] for s in stations)))
    for i in v0 for j in customers_prime if i != j
)

evrp.addConstrs(
    (y[j] <= YYijs[i, j, s] - hh[s, j] * distance[s, j] +
     (Q + mx_consump * np.max(distance)) * (1 - z[i, j, s]))
    for i in v0 for j in customers_prime for s in stations if i != j
)

evrp.addConstrs(
    (yijs[i, j, s] <= y[i] - hh[i, s] * distance[i, s] +
     (Q + mx_consump * np.max(distance)) * (1 - z[i, j, s]))
    for i in v0 for j in customers_prime for s in stations if i != j
)

evrp.addConstrs(
    (yijs[i, j, s] <= YYijs[i, j, s])
    for i in v0 for j in customers_prime for s in stations if i != j
)

evrp.addConstrs(
    (YYijs[i, j, s] <= z[i, j, s] * Q)
    for i in v0 for j in customers_prime for s in stations if i != j
)

# SOC bounds
evrp.addConstrs((y[i] <= Q) for i in v0)
evrp.addConstrs((y[i] <= Q) for i in vpr0)
evrp.addConstrs((y[i] >= 0) for i in v0)
evrp.addConstrs((YYijs[i, j, s] >= 0) for i in v0 for j in customers_prime for s in stations)
evrp.addConstrs((yijs[i, j, s] >= 0) for i in v0 for j in customers_prime for s in stations)

# station usage restrictions for dummy depots
evrp.addConstrs((z[i, j, num_v + nc] == 0) for i in v0 for j in dumm_depot_arrival)
evrp.addConstrs((z[i, j, num_v + nc] == 0) for i in dumm_depot_departure for j in customers_prime)

# dummy depots: exactly one outgoing / incoming
evrp.addConstrs((quicksum(x[i, j] for j in customers_prime) == 1)
                for i in dumm_depot_departure)
evrp.addConstrs((quicksum(x[i, j] for i in v0) == 1)
                for j in dumm_depot_arrival)

# ---------------- Connectivity cuts from old MILP3 ----------------

if cutss == 1:
    evrp.addConstrs((x[i, j] <= connect_cuts[i, j]) for (i, j) in cust_sets)
    evrp.addConstrs(
        (z[i, j, s] <= stat_dominators[i, j, s - nc - len(dumm_depot_departure) + 1])
        for (i, j) in cust_sets for s in stations
    )
    evrp.addConstrs(
        (x[i, j] - quicksum(z[i, j, s] for s in new_cut_sets[i, j]) == 0)
        for (i, j) in customer_cuts_set
    )


# ---------------- Objective: total charge + energy ----------------

evrp.setObjective(
    quicksum(y[i] for i in dumm_depot_departure) -
    quicksum(y[i] for i in dumm_depot_arrival) +
    quicksum(YYijs[i, j, s] - yijs[i, j, s]
             for i in v0 for j in customers_prime for s in stations if i != j),
    GRB.MINIMIZE
)

evrp.setParam("OutputFlag", 0)
evrp.setParam("TimeLimit", 7200)
evrp.setParam("MIPGap", 0.000001)

evrp.update()
evrp.optimize()

# =============================================================================
#                           PRINTING / POST-PROCESS
# =============================================================================

status = evrp.status
if status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    objVal = evrp.objVal
else:
    objVal = float('inf')

if write_in_text == 1:
    file.write('\n')
    if status == GRB.OPTIMAL:
        file.write('status: Optimal, status_code: ' + str(status) + '\n')
        file.write('OFV:  ' + str(evrp.objVal) + '\n')
    elif status == GRB.TIME_LIMIT:
        file.write('status: UB, status_code: ' + str(status) + '\n')
        file.write('Best_Bound:  ' + str(evrp.objVal) + '\n')
        file.write('GAP (%):  ' + str(100 * evrp.MIPgap) + '%\n')
    else:
        file.write('status: NOT Optimal, status_code: ' + str(status) + '\n')

    file.write('\nRun_time = ' + str(evrp.Runtime) + '\n')
    file.write('Total_time = ' + str(time.time() - start_time) + '\n\n')
else:
    print('\nRun_time = ' + str(evrp.Runtime))
    print('Total_time = ' + str(time.time() - start_time))
    print('objVal: ', objVal)
    print('num_veh: ', num_v)
    print('status:', status)
# ---------------- Extract solution x,z into routes ----------------

vehicles_copy = []
error_status = False

if status not in [GRB.INFEASIBLE, GRB.UNBOUNDED]:
    # --- Build successor mapping from x ---
    succ = {}
    for i in v0:
        for j in customers_prime:
            if x[i, j].X > 0.5:
                succ[i] = j
                break  # each i in v0 has at most one outgoing arc

    # --- Build routes starting from each dummy departure ---
    prt_solution = []
    for dep in dumm_depot_departure:
        if dep not in succ:
            continue  # unused vehicle
        route = [dep]
        curr = dep
        while True:
            nxt = succ.get(curr, None)
            if nxt is None:
                break
            route.append(nxt)
            if nxt in dumm_depot_arrival:
                break
            curr = nxt
        prt_solution.append(np.array(route, dtype=int))

    # --- Insert stations according to z (i -> s -> j) ---
    routes_with_stations = []
    for route in prt_solution:
        r_nodes = [int(route[0])]
        for idx in range(len(route) - 1):
            i = int(route[idx])
            j = int(route[idx + 1])

            # check if there is a station between i and j
            inserted = False
            for s in stations:
                if z[i, j, s].X > 0.5:
                    # path is i -> s -> j
                    r_nodes.append(int(s))
                    inserted = True
                    # assuming at most one station per (i,j); break is safe
                    break
            r_nodes.append(j)

        routes_with_stations.append(r_nodes)

    vehicles_copy = [{'index': r} for r in routes_with_stations]

else:
    vehicles_copy = []
    error_status = True

# ---------------- Post-evaluation of OFV from solution variables ----------------

if not error_status:
    # Objective is: sum(y at dummy departures)
    #             - sum(y at dummy arrivals)
    #             + sum(YYijs - yijs) over all station uses
    start_energy = sum(y[i].X for i in dumm_depot_departure)
    end_energy   = sum(y[i].X for i in dumm_depot_arrival)

    station_energy = 0.0
    for i in v0:
        for j in customers_prime:
            for s in stations:
                station_energy += YYijs[i, j, s].X - yijs[i, j, s].X

    ofvv = start_energy - end_energy + station_energy

    print(vehicles_copy)
    print('\n---------------------------------')
    print('OFV (post-evaluated) = ', ofvv)
    print('Model objVal         = ', objVal)
    print('Difference (post - model) = ', ofvv - objVal)

    if write_in_text == 1:
        file.write('\n-------------------------------------\n')
        file.write('OFV (post-evaluated): ' + str(ofvv) + '\n')
        file.write('\n' + str(vehicles_copy))

if write_in_text == 1 and file is not None:
    file.close()


if write_in_text == 1 and file is not None:
    file.close()

gc.collect()
