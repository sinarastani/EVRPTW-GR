# -*- coding: utf-8 -*-
"""
EVRPTW with gradient- and load-dependent energy consumption + energy-based cuts.
EVRPTW-GR 
Model 2 - linearized


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

    # 2) More general scan: any cell string that has '/' and is on a row mentioning `key`
    for _, row in df.iterrows():
        # if there is a StringID column, require it to match or contain the key
        if "StringID" in df.columns:
            sid = row["StringID"]
            if isinstance(sid, str) and key not in sid:
                # row is probably not about this parameter
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

    # 3) If we get here, the file simply does not contain that parameter
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
    # ---------- customer size ----------
    m = re.search(r'C(\d+)', filename)  # old naming with 'Ck'
    if m:
        n_cust = int(m.group(1))
    else:
        # new naming with '_50_21_'
        m2 = re.search(r'_(\d+)_\d+_', filename)
        if not m2:
            raise ValueError(f"Cannot infer customer size from filename: {filename}")
        n_cust = int(m2.group(1))

    # ---------- network group + size folder ----------
    if n_cust in (5, 10, 15):
        network_group = "Small_Network"
    elif n_cust == 25:
        network_group = "Medium_Network"
    elif n_cust in (50, 100):
        network_group = "Large_Network"
    else:
        raise ValueError(f"Unsupported customer size: {n_cust} in filename: {filename}")

    size_folder = f"{n_cust}_Customers"

    # ---------- slope folder ----------
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
# BASIC SETTINGS
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

# -------------------------------------------------------------------------
grad_mult   = 1.0       # scaling for altitude
h           = 1.0
truncation  = 0         # 0 = Euclidean, 1 = truncated * 100
write_in_text = 0       # 1 to write to text file (kept for compatibility)

hhh = str(h).replace(".", ",")

# =============================================================================
# READ INSTANCE DATA (NEW _NL FORMAT WITH ALTITUDE)
# =============================================================================



filepath = build_instance_path(filename)
data = read_data_file(filepath)
if data is None:
    raise RuntimeError(f"Data file could not be loaded: {filepath}")
    
# Separate node rows (d, f, c) from parameter rows (Q, C, r, g, v, ...)
nodes = data[data["Type"].isin(["d", "f", "c"])].reset_index(drop=True)

# Counts
ns = int(sum(nodes["Type"] == "f"))  # number of stations
nc = int(sum(nodes["Type"] == "c"))  # number of customers

# Parameters from bottom lines


Q = get_param_value(data, "Q")          # vehicle battery capacity
raw_vehicle_cap = get_param_value(data, "C")   # original load capacity (e.g. 200.0)
g = get_param_value(data, "g")          # inverse refuelling rate
v_speed = get_param_value(data, "v")    # average velocity (should be 1.0 in your files)

# IMPORTANT:
# In the data files, demands are already scaled to the 3650-capacity space.

vehicle_cap = 3650.0

# ------------------ Build time windows, demands, coordinates, altitude -----
# Node order: depot (0), customers (1..nc), stations (nc+1..nc+ns), depot-end (nc+ns+1)
# Original rows in 'nodes':
#   0       : depot
#   1..ns   : stations
#   ns+1..ns+nc : customers
#
# We reorder to [depot, customers, stations, depot] for all vectors.

# Time windows
early_arrive = (
    list(nodes.loc[0:0, "ReadyTime"].astype(float)) +
    list(nodes.loc[1 + ns:nc + ns, "ReadyTime"].astype(float)) +
    list(nodes.loc[1:ns, "ReadyTime"].astype(float)) +
    list(nodes.loc[0:0, "ReadyTime"].astype(float))
)

late_arrive = (
    list(nodes.loc[0:0, "DueDate"].astype(float)) +
    list(nodes.loc[1 + ns:nc + ns, "DueDate"].astype(float)) +
    list(nodes.loc[1:ns, "DueDate"].astype(float)) +
    list(nodes.loc[0:0, "DueDate"].astype(float))
)

service_time = (
    list(nodes.loc[0:0, "ServiceTime"].astype(float)) +
    list(nodes.loc[1 + ns:nc + ns, "ServiceTime"].astype(float)) +
    list(nodes.loc[1:ns, "ServiceTime"].astype(float)) +
    list(nodes.loc[0:0, "ServiceTime"].astype(float))
)

# Demands (already scaled in _NL files)
demand = (
    list(nodes.loc[0:0, "demand"].astype(float)) +
    list(nodes.loc[1 + ns:nc + ns, "demand"].astype(float)) +
    list(nodes.loc[1:ns, "demand"].astype(float)) +
    list(nodes.loc[0:0, "demand"].astype(float))
)

# Coordinates
cordinate_x = (
    list(nodes.loc[0:0, "x"].astype(float)) +
    list(nodes.loc[1 + ns:nc + ns, "x"].astype(float)) +
    list(nodes.loc[1:ns, "x"].astype(float)) +
    list(nodes.loc[0:0, "x"].astype(float))
)

cordinate_y = (
    list(nodes.loc[0:0, "y"].astype(float)) +
    list(nodes.loc[1 + ns:nc + ns, "y"].astype(float)) +
    list(nodes.loc[1:ns, "y"].astype(float)) +
    list(nodes.loc[0:0, "y"].astype(float))
)

# Altitude (now from the same file)
altitude = (
    list(nodes.loc[0:0, "altitude"].astype(float)) +
    list(nodes.loc[1 + ns:nc + ns, "altitude"].astype(float)) +
    list(nodes.loc[1:ns, "altitude"].astype(float)) +
    list(nodes.loc[0:0, "altitude"].astype(float))
)

# Apply gradient multiplier
altitude = [grad_mult * a for a in altitude]

total_node = nc + ns + 2
distance = np.zeros((total_node, total_node))

# DEMAND NORMALISATION (DISABLED: data FILES ALREADY SCALED)
# demand = [(3650 / raw_vehicle_cap) * d for d in demand]
# vehicle_cap = 3650.0

# =============================================================================
# DISTANCE & TRAVEL TIME
# =============================================================================

if truncation == 1:
    for i in range(total_node):
        for j in range(total_node):
            dist_temp = np.hypot(
                cordinate_x[i] - cordinate_x[j],
                cordinate_y[i] - cordinate_y[j],
            )
            if 100 * dist_temp - int(100 * dist_temp) < 1e-9:
                distance[i, j] = 100 * dist_temp
            else:
                distance[i, j] = int(100 * dist_temp) + 1
    travel_time = distance / 100.0 * v_speed
    Q = 100 * Q
    g = g / 100.0
else:
    for i in range(total_node):
        for j in range(total_node):
            distance[i, j] = np.hypot(
                cordinate_x[i] - cordinate_x[j],
                cordinate_y[i] - cordinate_y[j],
            )
    travel_time = distance / v_speed

if write_in_text == 1:
    result_file_name = f"Result_{hhh}_{filename}"
    file = open("Gradient\\" + result_file_name, "w")
    file.write("File_Name: " + filename + "\n")
    file.write("#Veh: " + str(num_v) + "\n")

# =============================================================================
#                             RANGES AND EXPRESSIONS
# =============================================================================

late_arrive0 = late_arrive[0]
customers = range(1, 1 + nc)           # 1..nc
cust = tuple(customers)

stations = range(1 + nc, 1 + nc + ns)  # nc+1 .. nc+ns
v_total = range(0, total_node)         # 0..total_node-1

customers_prime = ()
v0 = ()
dumm_depot_arrival = ()
dumm_depot_departure = ()

for i in range(1 + nc + ns, total_node):
    dumm_depot_arrival += (i,)
    dumm_depot_departure += (i - 1 - nc - ns,)

v0 = dumm_depot_departure + cust
customers_prime = cust + dumm_depot_arrival
vpr0 = v0 + dumm_depot_arrival

thatijs = np.zeros([total_node, total_node, ns + 1])
for i in range(total_node):
    for j in range(total_node):
        for s in stations:
            thatijs[i, j, s - nc - 1] = (
                travel_time[i, s] + travel_time[s, j] - travel_time[i, j]
            )

stat_tuple = tuple(stations)
v00 = v0 + stat_tuple
vpr00 = cust + stat_tuple + dumm_depot_arrival

# =============================================================================
# ENERGY PARAMETERS
# =============================================================================

weight = 6350.0
load = 0.0
vv = 60.0            # km/h
v_s = vv / 3.6       # m/s
gg = 9.81
cr = 0.01
cd = 0.7
ro = 1.2041
a = 3.912
kk = 1.0
conv_f = 1.0
KNV = 0.0
p_acc = 0.0
degree = 0.0
acceleration = 0.0

mu = 0.9
mu_train = 0.9
regenerating = 0.8

# =============================================================================
# GRADIENTS (using altitude from same file)
# =============================================================================

# Ensure depot start/end altitudes are 0 (as in previous code)
altitude[0] = 0.0
altitude[total_node - 1] = 0.0

gradient_list = []
angle = np.zeros([len(distance), len(distance)])
gradient = np.zeros([len(distance), len(distance)])
angle_list = []

for i in range(len(distance)):
    for j in range(len(distance)):
        if distance[i, j] == 0:
            angle[i, j] = 0.0
            gradient[i, j] = 0.0
        else:
            angle[i, j] = np.degrees(
                np.arctan((altitude[j] - altitude[i]) / distance[i, j])
            )
            gradient[i, j] = 100.0 * (altitude[j] - altitude[i]) / distance[i, j]
            angle_list.append(angle[i, j])
            gradient_list.append((altitude[j] - altitude[i]) / distance[i, j])

print("Average angle: ", np.average(np.abs(angle)))
print("average angle accurate", np.average(np.abs(angle_list)))
print("average gradient (%)", 100 * np.average(np.abs(gradient_list)))
print("---------------------\n")

# =============================================================================
# BASE CONSUMPTION (scaled)
# =============================================================================

p_tract_base = (
    (weight + load) * acceleration
    + (weight + load) * gg * np.sin(np.deg2rad(degree))
    + 0.5 * cd * ro * a * v_s * v_s
    + (weight + load) * gg * cr * np.cos(np.deg2rad(degree))
) * v_s / 1000.0

p_total_base = (KNV + (p_tract_base / mu_train + p_acc) / mu) / (kk * conv_f)
consump_base = p_total_base / vv

p_tract = (
    (weight + load) * acceleration
    + (weight + load) * gg * np.sin(np.deg2rad(angle))
    + 0.5 * cd * ro * a * v_s * v_s
    + (weight + load) * gg * cr * np.cos(np.deg2rad(angle))
) * v_s / 1000.0

p_total = np.zeros_like(p_tract)
for i in range(len(p_tract)):
    for j in range(len(p_tract)):
        if p_tract[i, j] > 0:
            p_total[i, j] = (KNV + (p_tract[i, j] / mu_train + p_acc) / mu) / (kk * conv_f)
        else:
            p_total[i, j] = regenerating * (KNV + (p_tract[i, j] + p_acc)) / (kk * conv_f)

consump = p_total / vv

extra_load = np.sum(demand)
p_tract_max = (
    (weight + extra_load) * acceleration
    + (weight + extra_load) * gg * np.sin(np.deg2rad(np.max(angle)))
    + 0.5 * cd * ro * a * v_s * v_s
    + (weight + extra_load) * gg * cr * np.cos(np.deg2rad(np.max(angle)))
) * v_s / 1000.0

p_total_max = (KNV + (p_tract_max / mu_train + p_acc) / mu) / (kk * conv_f)
consump_max = p_total_max / vv
mx_consump = consump_max / consump_base

# =============================================================================
# CUTS: ENERGY MIN/MAX ON ARCS (unchanged)
# =============================================================================

def min_energy_consumption(s, i):
    energy_consump_list = []
    for load in [0, sum(demand)]:
        p_tract_si = (
            (weight + load) * acceleration
            + (weight + load) * gg * np.sin(np.deg2rad(angle[s, i]))
            + 0.5 * cd * ro * a * v_s * v_s
            + (weight + load) * gg * cr * np.cos(np.deg2rad(angle[s, i]))
        ) * v_s / 1000.0
        if p_tract_si >= 0:
            p_total_si = (KNV + (p_tract_si / mu_train + p_acc) / mu) / (kk * conv_f)
            consump_si = p_total_si / vv
            energy_consump = consump_si / consump_base
        else:
            p_total_si = regenerating * (KNV + (p_tract_si + p_acc)) / (kk * conv_f)
            consump_si = p_total_si / vv
            energy_consump = consump_si / consump_base
        energy_consump_list.append(energy_consump)
    return min(energy_consump_list)


def max_energy_consumption(s, i):
    energy_consump_list = []
    for load in [0, sum(demand)]:
        p_tract_si = (
            (weight + load) * acceleration
            + (weight + load) * gg * np.sin(np.deg2rad(angle[s, i]))
            + 0.5 * cd * ro * a * v_s * v_s
            + (weight + load) * gg * cr * np.cos(np.deg2rad(angle[s, i]))
        ) * v_s / 1000.0
        if p_tract_si >= 0:
            p_total_si = (KNV + (p_tract_si / mu_train + p_acc) / mu) / (kk * conv_f)
            consump_si = p_total_si / vv
            energy_consump = consump_si / consump_base
        else:
            p_total_si = regenerating * (KNV + (p_tract_si + p_acc)) / (kk * conv_f)
            consump_si = p_total_si / vv
            energy_consump = consump_si / consump_base
        energy_consump_list.append(energy_consump)
    return max(energy_consump_list)


def connectivity_cuts(v):
    stat_dominators = np.ones([2 + nc + ns, 2 + nc + ns, ns + 1])
    dominant_stations_dic = {}

    # -------- station dominance -------------------------------------
    for i in range(2 + ns + nc):
        if nc < i < 1 + nc + ns:
            continue
        for j in range(2 + ns + nc):
            if nc < j < 1 + nc + ns:
                continue
            if i == j:
                continue
            if (i == 0 and j == 1 + ns + nc) or (i == 1 + ns + nc and j == 0):
                continue

            dominant = list(stations)
            for k in range(1 + nc, 1 + nc + ns):
                for s in list(dominant):
                    if k != s:
                        if (distance[i, k] * max_energy_consumption(i, k)
                                <= distance[i, s] * min_energy_consumption(i, s)
                                and distance[k, j] * max_energy_consumption(k, j)
                                <= distance[s, j] * min_energy_consumption(s, j)):
                            dominant.remove(s)
            dominant_stations_dic[i, j] = dominant

    # -------- max possible SOC at each node ------------------------
    max_soc = []
    for i in range(0, 2 + ns + nc):
        possible_dep_soc = []
        for s in stations:
            possible_dep_soc.append(Q - distance[s, i] * min_energy_consumption(s, i))
        max_soc.append(max(possible_dep_soc))

    # -------- minimum required SOC at each node --------------------
    least_soc = []
    for i in range(0, 2 + ns + nc):
        possible_arr_soc = []
        for s in stations:
            possible_arr_soc.append(distance[s, i] * min_energy_consumption(i, s))
        least_soc.append(max(0, min(possible_arr_soc)))

    # -------- earliest feasible arrival time at each node ----------
    earliest_possible = []
    for i in range(0, 2 + ns + nc):
        earliest_possible.append(max(early_arrive[i], travel_time[0, i]))

    # -------- main connectivity check ------------------------------
    connect_cuts = np.zeros(np.shape(distance))
    cust_sets = []
    new_cut_sets = {}

    for i in range(0, 2 + ns + nc):
        if nc < i <= nc + ns:
            continue
        for j in range(0, 2 + ns + nc):
            if nc < j <= nc + ns or i == j:
                continue

            # quick infeasibility checks
            if (demand[i] + demand[j] > vehicle_cap or
                earliest_possible[i] > late_arrive[i] or
                earliest_possible[i] + service_time[i] + travel_time[i, j] > late_arrive[j] or
                max(earliest_possible[i] + service_time[i] + travel_time[i, j],
                    early_arrive[j]) + service_time[j] + travel_time[j, total_node - 1]
                    > late_arrive[total_node - 1]):
                continue
            else:
                # direct feasible?
                if (max_soc[i]
                        - min_energy_consumption(i, j) * distance[i, j]
                        - least_soc[j] >= 0):
                    connect_cuts[i, j] = 1
                    if j != 0 and i != nc + ns + 1:
                        cust_sets.append((i, j))
                else:
                    # try via station(s)
                    for k in dominant_stations_dic[i, j]:
                        if (max_soc[i] - min_energy_consumption(i, k) * distance[i, k] >= 0 and
                            min_energy_consumption(k, j) * distance[k, j] + least_soc[j] <= Q):
                            recharge = (min_energy_consumption(k, j) * distance[k, j]
                                        + least_soc[j]
                                        - (max_soc[i]
                                           - min_energy_consumption(i, k) * distance[i, k]))
                            if (recharge <= Q and
                                (max_soc[i] - min_energy_consumption(i, k) * distance[i, k])
                                + recharge <= Q):

                                # time feasibility with recharge
                                if (earliest_possible[i] + service_time[i] + travel_time[i, k]
                                        <= late_arrive[k]
                                    and max(earliest_possible[i] + service_time[i]
                                            + travel_time[i, k], early_arrive[k])
                                    + recharge * g + travel_time[k, j] <= late_arrive[j]
                                    and max(max(earliest_possible[i] + service_time[i]
                                                + travel_time[i, k], early_arrive[k])
                                            + recharge * g + travel_time[k, j],
                                            early_arrive[j])
                                    + travel_time[j, total_node - 1]
                                    <= late_arrive[total_node - 1]):

                                    connect_cuts[i, j] = 1
                                    if j != 0 and i != nc + ns + 1:
                                        cust_sets.append((i, j))
                                    if i != nc + ns + 1:
                                        new_cut_sets[i, j] = dominant_stations_dic[i, j]
                                    break

    return (connect_cuts, stat_dominators, new_cut_sets, cust_sets)


(connect_cuts, stat_dominators, new_cut_sets, cust_sets) = connectivity_cuts(v_speed)
stat_dominators[:, :, 0] = 0

customer_cuts_set = list(new_cut_sets.keys())
for idx, (i_val, j_val) in enumerate(customer_cuts_set):
    if j_val == 0:
        customer_cuts_set[idx] = (i_val, nc + ns + 1)

LB_Veh = np.ceil(np.sum(demand) / vehicle_cap)

# =============================================================================
#                             OPTIMIZATION MODEL
# =============================================================================

start_time = time.time()
evrp = Model("EVRP")

# ------------------------ Decision Variables -----------------------------

t = evrp.addVars(v_total, lb=0, vtype=GRB.CONTINUOUS, name="t")

z = evrp.addVars(v0, customers_prime, stations, lb=0, vtype=GRB.BINARY, name="z")
x = evrp.addVars(v0, customers_prime, lb=0, vtype=GRB.BINARY, name="x")

y = evrp.addVars(vpr0, lb=0, vtype=GRB.CONTINUOUS, name="y")
YYijs = evrp.addVars(v0, customers_prime, stations, lb=0, vtype=GRB.CONTINUOUS, name="YYijs")
yijs = evrp.addVars(v0, customers_prime, stations, lb=0, vtype=GRB.CONTINUOUS, name="yijs")

hh = evrp.addVars(v00, vpr00, lb=-Q, vtype=GRB.CONTINUOUS, name="hh")
zz = evrp.addVars(v00, vpr00, lb=0, vtype=GRB.BINARY, name="zz")
ptractive = evrp.addVars(v00, vpr00, lb=-Q, vtype=GRB.CONTINUOUS, name="ptractive")

u = evrp.addVars(v00, vpr00, lb=0, vtype=GRB.CONTINUOUS, name="u")
uu = evrp.addVars(v00, vpr00, stations, lb=0, vtype=GRB.CONTINUOUS, name="uu")

# ------------------------ Constraints ------------------------------------

# Load and capacity without stations
evrp.addConstrs(
    (u[i, j] == weight * x[i, j])
    for i in dumm_depot_departure for j in customers_prime
)

evrp.addConstrs(
    (demand[i] * x[i, j] + weight * x[i, j] <= u[i, j])
    for i in customers for j in customers_prime if i != j
)

evrp.addConstrs(
    (u[i, j] <= (weight + np.sum(demand)) * x[i, j])
    for i in customers for j in customers_prime
)

evrp.addConstrs(
    (u[j, k] >= quicksum(u[i, j] for i in v0)
     + demand[j] * x[j, k]
     - (weight + vehicle_cap) * (1 - x[j, k]))
    for j in customers for k in customers_prime
)

evrp.addConstrs(
    (u[j, k] <= quicksum(u[i, j] for i in v0)
     + demand[j] * x[j, k]
     + (weight + vehicle_cap) * (1 - x[j, k]))
    for j in customers for k in customers_prime
)

# Load and capacity with stations
evrp.addConstrs(
    (uu[i, j, s] == weight * z[i, j, s])
    for i in dumm_depot_departure for j in customers_prime for s in stations
)

evrp.addConstrs(
    (uu[j, k, s] >= quicksum(u[i, j] for i in v0)
     + demand[j] * z[j, k, s]
     - (weight + vehicle_cap) * (1 - z[j, k, s]))
    for j in customers for k in customers_prime for s in stations
)

evrp.addConstrs(
    (uu[j, k, s] <= quicksum(u[i, j] for i in v0)
     + demand[j] * z[j, k, s]
     + (weight + vehicle_cap) * (1 - z[j, k, s]))
    for j in customers for k in customers_prime for s in stations
)

evrp.addConstrs(
    (uu[i, j, s] <= z[i, j, s] * (np.sum(demand) + weight))
    for i in customers for j in customers_prime for s in stations
)

# Routing and time
evrp.addConstrs(
    (quicksum(x[i, j] for j in customers_prime if i != j) == 1)
    for i in customers
)

evrp.addConstrs(
    (quicksum(x[i, j] for i in v0 if i != j)
     - quicksum(x[j, i] for i in customers_prime if i != j) == 0)
    for j in customers
)

evrp.addConstrs(
    (quicksum(z[i, j, s] for s in stations) <= x[i, j])
    for i in v0 for j in customers_prime
)

evrp.addConstrs(
    (t[i]
     + x[i, j] * (travel_time[i, j] + service_time[i])
     + quicksum(z[i, j, s] * thatijs[i, j, s - nc - 1]
                + g * (YYijs[i, j, s] - yijs[i, j, s])
                for s in stations)
     - late_arrive0 * (1 - x[i, j])
     <= t[j])
    for i in v0 for j in customers_prime if i != j
)

evrp.addConstrs((t[j] >= early_arrive[j]) for j in v_total)
evrp.addConstrs((t[j] <= late_arrive[j]) for j in v_total)

# Energy / ptractive expressions
evrp.addConstrs(
    (ptractive[i, j] >=
     ((KNV + (( (u[i, j] - quicksum(uu[i, j, s] for s in stations)) * acceleration
                + (u[i, j] - quicksum(uu[i, j, s] for s in stations))
                  * gg * np.sin(np.deg2rad(angle[i, j]))
                + (x[i, j] - quicksum(z[i, j, s] for s in stations))
                  * 0.5 * cd * ro * a * v_s * v_s
                + (u[i, j] - quicksum(uu[i, j, s] for s in stations))
                  * gg * cr * np.cos(np.deg2rad(angle[i, j])))
               * v_s / 1000.0))
      / (kk * conv_f)) / vv)
    for i in v0 for j in customers_prime if i != j
)

evrp.addConstrs(
    (ptractive[s, j] >=
     ((KNV + ((quicksum(uu[i, j, s] for i in v0) * acceleration
               + quicksum(uu[i, j, s] for i in v0)
                 * gg * np.sin(np.deg2rad(angle[s, j]))
               + quicksum(z[i, j, s] for i in v0)
                 * 0.5 * cd * ro * a * v_s * v_s
               + quicksum(uu[i, j, s] for i in v0)
                 * gg * cr * np.cos(np.deg2rad(angle[s, j])))
              * v_s / 1000.0))
      / (kk * conv_f)) / vv)
    for j in customers_prime for s in stations
)

evrp.addConstrs(
    (ptractive[i, s] >=
     ((KNV + ((quicksum(uu[i, j, s] for j in customers_prime) * acceleration
               + quicksum(uu[i, j, s] for j in customers_prime)
                 * gg * np.sin(np.deg2rad(angle[i, s]))
               + quicksum(z[i, j, s] for j in customers_prime)
                 * 0.5 * cd * ro * a * v_s * v_s
               + quicksum(uu[i, j, s] for j in customers_prime)
                 * gg * cr * np.cos(np.deg2rad(angle[i, s])))
              * v_s / 1000.0))
      / (kk * conv_f)) / vv)
    for i in v0 for s in stations if i != s
)

evrp.addConstrs((Q * zz[i, j] >= ptractive[i, j])
                for i in v00 for j in vpr00 if i != j)
evrp.addConstrs((Q * (zz[i, j] - 1) <= ptractive[i, j])
                for i in v00 for j in vpr00 if i != j)

evrp.addConstrs(
    (hh[i, j] >= Q * (zz[i, j] - 1)
     + (ptractive[i, j] / (mu_train * mu)) / consump_base)
    for i in v00 for j in vpr00 if i != j
)

evrp.addConstrs(
    (hh[i, j] >= -Q * zz[i, j]
     + regenerating * ptractive[i, j] / consump_base)
    for i in v00 for j in vpr00 if i != j
)

# Force hh = 0 on station-to-station arcs
evrp.addConstrs((hh[i, j] == 0) for i in stations for j in stations)

# SOC constraints
evrp.addConstrs(
    (y[j] <= y[i]
     - hh[i, j] * distance[i, j]
     + (Q + mx_consump * np.max(distance))
       * (1 - x[i, j] + quicksum(z[i, j, s] for s in stations)))
    for i in v0 for j in customers_prime if i != j
)

evrp.addConstrs(
    (y[j] <= YYijs[i, j, s]
     - hh[s, j] * distance[s, j]
     + (Q + mx_consump * np.max(distance)) * (1 - z[i, j, s]))
    for i in v0 for j in customers_prime for s in stations if i != j
)

evrp.addConstrs(
    (yijs[i, j, s] <= y[i]
     - hh[i, s] * distance[i, s]
     + (Q + mx_consump * np.max(distance)) * (1 - z[i, j, s]))
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

evrp.addConstrs((y[i] <= Q) for i in v0)
evrp.addConstrs((y[i] <= Q) for i in vpr0)
evrp.addConstrs((y[i] >= 0) for i in v0)
evrp.addConstrs((YYijs[i, j, s] >= 0)
                for i in v0 for j in customers_prime for s in stations)
evrp.addConstrs((yijs[i, j, s] >= 0)
                for i in v0 for j in customers_prime for s in stations)

# Logical constraints for z (no station between depot->depot)
evrp.addConstrs((z[i, j, 1 + nc] == 0)
                for i in v0 for j in dumm_depot_arrival)
evrp.addConstrs((z[i, j, 1 + nc] == 0)
                for i in dumm_depot_departure for j in customers_prime)

# Fleet size
evrp.addConstr(
    (quicksum(x[i, j] for i in dumm_depot_departure for j in customers) == num_v)
)
evrp.addConstr(
    (quicksum(x[i, j] for i in customers for j in dumm_depot_arrival) == num_v)
)

# Connectivity cuts
evrp.addConstrs((x[i, j] <= connect_cuts[i, j]) for (i, j) in cust_sets)
evrp.addConstrs((z[i, j, s] <= stat_dominators[i, j, s - nc])
                for (i, j) in cust_sets for s in stations)
evrp.addConstrs(
    (x[i, j] - quicksum(z[i, j, s] for s in new_cut_sets[i, j]) == 0)
    for (i, j) in customer_cuts_set
)


# Objective
evrp.setObjective(
    quicksum(hh[i, j] * distance[i, j] for i in v00 for j in vpr00 if i != j),
    GRB.MINIMIZE,
)

evrp.setParam("OutputFlag", 0)
evrp.setParam("TimeLimit", 7200)
evrp.setParam("MIPGap", 0.000001)

evrp.update()
evrp.optimize()

status = evrp.status
if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
    objVal = evrp.objVal
else:
    objVal = float("inf")

if write_in_text == 1:
    file.write("  \n")
    if status == GRB.OPTIMAL:
        file.write("status: Optimal, status_code: " + str(status) + "\n")
        file.write("OFV:  " + str(evrp.objVal) + "\n")
    elif status == GRB.TIME_LIMIT:
        file.write("status: UB, status_code: " + str(status) + "\n")
        file.write("Best_Bound:  " + str(evrp.objVal) + "\n")
        file.write("GAP (%):  " + str(100 * evrp.MIPgap) + "%\n")
    else:
        file.write("status: NOT Optimal, status_code: " + str(status) + "\n")
    file.write("  \n")
    file.write("Run_time = " + str(evrp.Runtime) + "\n")
    file.write("Total_time = " + str(time.time() - start_time) + "\n")
    file.write("  \n")
    file.write("Decision Variables >>>>>>>>\n\n")
else:
    print("\nRun_time = " + str(evrp.Runtime) + "\n")
    print("Total_time = " + str(time.time() - start_time) + "\n")
    print("objVal: ", objVal)
    print("num_veh: ", num_v)
    print("status", evrp.status)

# =============================================================================
# ROUTE RECONSTRUCTION 
# =============================================================================
# if status !=3:
#     for v in evrp.getVars(): 
#         if evrp.objVal < 1e+99  and v.x!=0:
#             if write_in_text == 0: # printing variables
#                 print('%s %f'%(v.Varname,v.x))



#-----
# =============================================================================
# ROUTE RECONSTRUCTION (first customers, then insert stations)
# =============================================================================

error_status = True
vehicles_copy = []

if status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
    # ---------- 1) build customer-only routes from x ----------
    vis = []
    Sol_x = np.zeros((len(v0), len(customers_prime)))

    for idx, i in enumerate(v0):
        for jdx, j in enumerate(customers_prime):
            if evrp.objVal < 1e99:
                Sol_x[idx, jdx] = x[i, j].X
            else:
                error_status = True
                ofvv = 1e99
            if 1 - 1e-5 <= Sol_x[idx, jdx] <= 1 + 1e-5:
                vis.append((i, j))

    if len(vis) == 0:
        print("No arcs with x=1; nothing to reconstruct.")
    else:
        # nodes that actually appear in solution
        nodes_used = np.unique([n for arc in vis for n in arc])

        visited = np.array(vis, dtype=int)
        prt_solution = []

        # reconstruct all routes (customer-only)
        while visited.size > 0:
            # pick a starting arc
            if visited[0, 0] in dumm_depot_departure:
                sol = [visited[0, 0], visited[0, 1]]
            elif visited[0, 1] in dumm_depot_departure:
                sol = [visited[0, 1], visited[0, 0]]
            else:
                # fallback if for some reason first arc has no depot
                sol = [visited[0, 0], visited[0, 1]]

            visited = np.delete(visited, 0, axis=0)

            # extend current route until dummy arrival depot
            while True:
                if sol[-1] in dumm_depot_arrival:
                    break
                # find arc whose tail == last node of sol
                idx_next = np.where(visited[:, 0] == sol[-1])[0]
                if idx_next.size == 0:
                    # try reversed arcs (should not happen in a clean solution)
                    idx_next = np.where(visited[:, 1] == sol[-1])[0]
                    if idx_next.size == 0:
                        break
                    nxt = visited[idx_next[0], 0]
                else:
                    nxt = visited[idx_next[0], 1]

                sol.append(int(nxt))
                visited = np.delete(visited, idx_next[0], axis=0)
                if visited.size == 0:
                    break

            prt_solution.append(np.array(sol, dtype=int))

        # ---------- 2) insert stations between consecutive customers ----------
        vehicles_with_stations = []
        chrgd = []

        for route_arr in prt_solution:
            base_route = list(route_arr)       # e.g. [0, 29, 28, 26, 25, 27, 72]
            new_route = [base_route[0]]        # start from the depot
            chrgd_route = [0.0]                # charge amount per node, same length as new_route

            for pos in range(len(base_route) - 1):
                i = base_route[pos]
                j = base_route[pos + 1]

                # check if any station is used on arc (i, j)
                inserted = False
                for s in stations:
                    if abs(z[i, j, s].X - 1.0) <= 1e-5:
                        # insert station between i and j
                        new_route.append(s)
                        chrgd_route.append(YYijs[i, j, s].X - yijs[i, j, s].X)
                        inserted = True

                # now append the original head node j
                new_route.append(j)
                chrgd_route.append(0.0)

            vehicles_with_stations.append({"index": new_route})
            chrgd.append(chrgd_route)

        vehicles_copy = vehicles_with_stations
        error_status = False
else:
    vehicles_copy = []
    error_status = True
    ofvv = 1e99


# =============================================================================
#                       POST-PROCESSING: ENERGY ALONG ROUTES
# =============================================================================

if not error_status and len(vehicles_copy) > 0:
    consumption_list = []
    h_lists = []

    for i in range(num_v):
        if i >= len(vehicles_copy):
            break
        consumption = []
        demand_route = 0.0
        for jdx, j in enumerate(vehicles_copy[i]["index"][:-1]):
            next_j = vehicles_copy[i]["index"][jdx + 1]
            demand_route += demand[j]
            p_tract_ij = (
                (weight + demand_route) * acceleration
                + (weight + demand_route) * gg * np.sin(np.deg2rad(angle[j, next_j]))
                + 0.5 * cd * ro * a * v_s * v_s
                + (weight + demand_route) * gg * cr * np.cos(np.deg2rad(angle[j, next_j]))
            ) * v_s / 1000.0
            if p_tract_ij > 0:
                p_total_ij = (KNV + (p_tract_ij / mu_train + p_acc) / mu) / (kk * conv_f)
            else:
                p_total_ij = regenerating * (KNV + (p_tract_ij + p_acc)) / (kk * conv_f)
            hh00 = p_total_ij / (vv * consump_base)
            ofv0 = hh00 * distance[j, next_j]
            consumption.append(ofv0)
            h_lists.append(hh00)
        consumption.append(0.0)
        consumption_list.append(consumption)

    chrge_amount = []
    for i in range(len(consumption_list)):
        ch_am = []
        chrg_temp1 = 0.0
        chrg_temp2 = 0.0
        for jdx, j in enumerate(reversed(vehicles_copy[i]["index"][0:])):
            temp0 = consumption_list[i][len(consumption_list[i]) - jdx - 1]
            chrg_temp1 += temp0
            if chrg_temp1 > 0:
                chrg_temp2 = chrg_temp1
            else:
                chrg_temp1 = 0.0
            if j > nc or (jdx == len(vehicles_copy[i]["index"]) - 1):
                ch_am.append(chrg_temp2)
                chrg_temp1 = 0.0
                chrg_temp2 = 0.0
            else:
                ch_am.append(0.0)
        ch_am.reverse()
        chrge_amount.append(ch_am)

    soc_tot = []
    org_chrg_list = [0.0]
    for i in range(len(consumption_list)):
        soc_route = [Q]
        for jdx, j in enumerate(vehicles_copy[i]["index"][:-1]):
            temp = soc_route[-1] - consumption_list[i][jdx]
            if chrge_amount[i][jdx + 1] > 0:
                org_chrg = chrge_amount[i][jdx + 1] - temp
                org_chrg_list.append(org_chrg)
            else:
                org_chrg = 0.0
                org_chrg_list.append(org_chrg)
            temp += org_chrg
            if temp > Q:
                temp = Q
            soc_route.append(temp)
        soc_tot.append(soc_route)

    ofvv = np.sum(org_chrg_list)
    for i in range(len(consumption_list)):
        if consumption_list[i][0] >= 0:
            ofvv += Q - soc_tot[i][-1]
        else:
            ofvv += Q + consumption_list[i][0] - soc_tot[i][-1]

    print(vehicles_copy)
    print()
    print("---------------------------------")
    print("OFV (post-processed) = ", ofvv)

    if write_in_text == 1:
        file.write("\n")
        file.write("-------------------------------------\n")
        file.write("OFV: " + str(ofvv) + "\n\n")
        file.write(str(vehicles_copy))
        file.close()

    tt_dist = 0.0
    for route in vehicles_copy:
        for jdx in range(len(route["index"]) - 1):
            tt_dist += distance[route["index"][jdx], route["index"][jdx + 1]]

    print("distance", tt_dist)

    reg_amount = 0.0
    for cons_list in consumption_list:
        for val in cons_list:
            if val < 0:
                reg_amount += abs(val)
    print("reg_amount: ", reg_amount)
    print("reg_percentage: ", 100 * reg_amount / Q)
else:
    print("No feasible routes reconstructed or model infeasible.")
