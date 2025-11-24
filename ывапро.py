"""
train_td_plotter.py

Time-Distance diagram + conflict resolver for single-track section with:
- blocks (l1..ln)
- per-block type: 'avto' or 'yarim'
- global CLEARANCE (minutes) for pass-through
- yarim wait (1 min after release)
- AVTO_I (minutes) minimal send-interval for 'avto' blocks
- siding (passing loops) at stations
- 3 priorities: passenger1 (non-stop), passenger2 (stops), freight (yields)
- accepted/passer rule: passer >= accepted.dep + CLEARANCE ; accepted resumes 1 min after passer cleared

Outputs:
- conflicts_report.csv
- time_distance.png (time-distance diagram)
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------------
# USER PARAMETERS (EDIT)
# -------------------------
# Blocks (lengths). Units: kilometers (km). There are N blocks and N+1 stations.
blocks_km = [2.0, 1.8, 2.5]   # l1, l2, l3, ...

# Stations in order: A, a, b, c, ... , B
stations = ['A', 'a', 'b', 'c', 'B']  # len = len(blocks_km)+1

# Which stations have siding (passing loop)? True/False per station name
siding_at = { 'A': False, 'a': True, 'b': True, 'c': False, 'B': False }

# Block types per block: either 'avto' or 'yarim' (len = len(blocks_km))
block_types = ['avto', 'yarim', 'avto']

# Global parameters (minutes)
CLEARANCE = 3.0        # minimal pass clearance (minutes) — YOU set this
AVTO_I   = 3.0         # for 'avto' blocks: minimal send interval between same-direction trains (minutes)
YARIM_WAIT = 1.0       # for 'yarim' blocks: wait after release (minutes)
SAFETY = 0.5           # extra margin for travel times (minutes)

# WAIT_AFTER_PASS for accepted train to resume after passer clears (minutes)
WAIT_AFTER_PASS = 1.0

# Train types: speeds km/h, dwell at stations (minutes)
train_types = {
    'passenger1': {'speed_kmph':140.0, 'dwell_min':0.0},  # non-stop (only A and B)
    'passenger2': {'speed_kmph':100.0, 'dwell_min':2.0},  # stops at each station
    'freight':     {'speed_kmph':60.0,  'dwell_min':0.0}   # freight
}

# Priority map (higher = higher priority)
priority_map = {'passenger1': 3, 'passenger2': 2, 'freight': 1}

# Example train list: if you have CSV loading, you can replace this.
# direction: 'A->B' or 'B->A'
# start_station: either 'A' or 'B' (we assume trains start at an end for simplicity)
# start_time: "HH:MM" local time
trains = [
    {'id':'P1','type':'passenger1','direction':'A->B','start_station':'A','start_time':'08:00'},
    {'id':'F1','type':'freight','direction':'B->A','start_station':'B','start_time':'08:05'},
    {'id':'P2','type':'passenger2','direction':'B->A','start_station':'B','start_time':'08:10'},
    {'id':'F2','type':'freight','direction':'A->B','start_station':'A','start_time':'08:12'},
    {'id':'P3','type':'passenger2','direction':'A->B','start_station':'A','start_time':'08:20'},
]

# Optional: path to image/map you uploaded (developer provided path)
SAMPLE_IMAGE = 'sandbox:/mnt/data/ec1ff9a1-7d6e-452c-a7d8-76fcf6490212.png'

# -------------------------
# HELPERS
# -------------------------
def hhmm_to_min(t):
    """Convert HH:MM string to minutes from midnight"""
    if t is None or t=="":
        return None
    hh, mm = map(int, str(t).split(':'))
    return hh*60 + mm

def min_to_hhmm(m):
    if m is None:
        return ""
    hh = int(m // 60) % 24
    mm = int(m % 60)
    return f"{hh:02d}:{mm:02d}"

def travel_time_block_min(block_km, train_type):
    v = train_types[train_type]['speed_kmph']
    t_min = (block_km / v) * 60.0
    return t_min + SAFETY

# Build stop lists per train (stations, times). For passenger1 (non-stop) we only include A and B.
def build_train_stops(train):
    ttype = train['type']
    direction = train['direction']
    if direction == 'A->B':
        route = stations
    else:
        route = list(reversed(stations))
    # starting station index
    start_idx = route.index(train['start_station'])
    # We'll compute times block by block starting from start_idx
    stops = []
    # entry time to start station is start_time (train ready time)
    t = hhmm_to_min(train['start_time'])
    # If starting at end and that is departure time from that end, it's the dep time at that station
    # For simplicity, treat that as arrival==dep==start_time (train ready)
    # Append stop at start station
    dwell = 0.0
    if ttype=='passenger2':
        dwell = train_types[ttype]['dwell_min']
    elif ttype=='passenger1':
        dwell = 0.0
    elif ttype=='freight':
        dwell = train_types[ttype]['dwell_min']
    stops.append({'station': route[start_idx], 'arr': t, 'dep': t + dwell, 'dwell': dwell})
    curr_time = t + dwell
    # traverse remaining stations
    for i in range(start_idx, len(route)-1):
        # block index correspond to position between stations in original order
        # If direction A->B and stations = [A,a,b,c,B], block indices 0,1,2 correspond
        # For B->A, block index mapping still matches absolute blocks:
        if direction == 'A->B':
            block_idx = i  # 0..N-1
        else:
            # reversed: if route reversed, block index is len(blocks)-1 - i'
            block_idx = len(blocks_km) - 1 - i
        block_km = blocks_km[block_idx]
        tt = travel_time_block_min(block_km, ttype)
        curr_time = curr_time + tt
        # arrival at next station
        next_station = route[i+1]
        # dwell: passenger2 stops at all stations; passenger1 only at A and B
        if ttype=='passenger2':
            dwell_next = train_types[ttype]['dwell_min']
        elif ttype=='passenger1':
            # stop only if next_station is an end (A or B)
            dwell_next = 0.0
            if next_station in (stations[0], stations[-1]):
                dwell_next = 0.0  # keep 0 unless you want to model end-of-line dwell
        else:
            dwell_next = train_types[ttype]['dwell_min']
        stops.append({'station': next_station, 'arr': curr_time, 'dep': curr_time + dwell_next, 'dwell': dwell_next})
        curr_time = curr_time + dwell_next
    return stops

# Build initial schedule: per train, list of entries per block with entry/release times
def build_intervals_from_schedule(trains):
    # Each interval: train_id, type, block_index (0..N-1), entry_time (min), release_time (min), direction
    intervals = []
    per_train_stops = {}
    for tr in trains:
        stops = build_train_stops(tr)
        per_train_stops[tr['id']] = stops
        # for each block traversed, build occupancy interval
        # if train goes A->B, blocks index 0..N-1
        direction = tr['direction']
        if direction == 'A->B':
            # stops[0] at station 0, stops[1] at station1 => block 0 between them
            for i in range(len(stops)-1):
                block_idx = i
                entry_time = stops[i]['dep']  # train leaves station i and enters block
                travel = travel_time_block_min(blocks_km[block_idx], tr['type'])
                # plus dwell? travel_time already includes SAFETY; release is entry + travel
                release_time = entry_time + travel
                intervals.append({'train_id': tr['id'], 'type': tr['type'], 'block': block_idx,
                                  'entry': entry_time, 'release': release_time, 'dir':'A->B'})
        else:
            # B->A: stops list is reversed so block indexing must map:
            # stops[0] at station B (index N), stops[1] at station N-1 => block N-1 between them
            for i in range(len(stops)-1):
                # global block index is len(blocks)-1 - i
                block_idx = len(blocks_km) - 1 - i
                entry_time = stops[i]['dep']
                travel = travel_time_block_min(blocks_km[block_idx], tr['type'])
                release_time = entry_time + travel
                intervals.append({'train_id': tr['id'], 'type': tr['type'], 'block': block_idx,
                                  'entry': entry_time, 'release': release_time, 'dir':'B->A'})
    intervals_df = pd.DataFrame(intervals)
    return intervals_df, per_train_stops

# Conflict detection according to block types, CLEARANCE, AVTO_I, YARIM_WAIT and siding rule
def detect_conflicts(intervals_df, per_train_stops):
    # We'll compare intervals per block
    conflicts = []
    # easier access per-train mapping of stops by station index
    for block_idx, group in intervals_df.groupby('block'):
        rows = group.sort_values('entry').to_dict('records')
        bt = block_types[block_idx]
        # find station name for "between" index: station_left = stations[block_idx], right = stations[block_idx+1]
        station_left = stations[block_idx]
        station_right = stations[block_idx+1]
        # passing loop is usually at either left or right station (we consider both)
        siding_exists = siding_at.get(station_left, False) or siding_at.get(station_right, False)
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                a = rows[i]; b = rows[j]
                # If both same direction:
                if a['dir'] == b['dir']:
                    # for 'avto' require entries separated by AVTO_I
                    if bt == 'avto':
                        if b['entry'] < a['entry'] + AVTO_I:
                            conflicts.append({'block':block_idx,'type':'avto_same_dir_I','a':a,'b':b})
                    else: # 'yarim'
                        # require b.entry >= a.release + YARIM_WAIT
                        if b['entry'] < a['release'] + YARIM_WAIT:
                            conflicts.append({'block':block_idx,'type':'yarim_same_dir_wait','a':a,'b':b})
                else:
                    # opposite directions: if siding exists near this block we may allow pass-with-rule
                    if siding_exists:
                        # Use accepted/passer logic later in resolver (not mark as terminal conflict now)
                        # but if times overlap badly (both inside block simultaneously) still a conflict
                        if not (a['release'] <= b['entry'] or b['release'] <= a['entry']):
                            # potential meet; mark for special handling
                            conflicts.append({'block':block_idx,'type':'opposite_meet_siding','a':a,'b':b, 'siding': True})
                    else:
                        # no siding => any overlap is conflict
                        if not (a['release'] <= b['entry'] or b['release'] <= a['entry']):
                            conflicts.append({'block':block_idx,'type':'opposite_meet_nosiding','a':a,'b':b, 'siding': False})
    return conflicts

# Greedy resolver implementing priorities and accepted/passer business
def greedy_resolve(trains, intervals_df, per_train_stops):
    # delays dict (minutes) applied to whole train schedule
    delays = {tr['id']: 0.0 for tr in trains}
    # We'll iterate: detect conflicts -> resolve first conflict -> apply delay -> rebuild intervals
    iteration = 0
    max_iter = 5000
    while True:
        iteration += 1
        intervals_df, per_train_stops = build_intervals_from_schedule(
            [dict(tr, start_time=min_to_hhmm(hhmm_to_min(tr['start_time']) + delays[tr['id']])) for tr in trains]
        )
        conflicts = detect_conflicts(intervals_df, per_train_stops)
        if not conflicts or iteration > max_iter:
            break
        # pick first conflict and resolve
        c = conflicts[0]
        a = c['a']; b = c['b']
        # determine priorities: if one is passenger and other freight => freight delays
        pri_a = priority_map[a['type']]
        pri_b = priority_map[b['type']]
        # For opposite meets with siding: apply accepted/passer rule:
        if c['type'] == 'opposite_meet_siding':
            # earlier entry gets accepted (stops) and later passes (doesn't stop) — but passenger vs freight rule overrides
            if a['entry'] <= b['entry']:
                accepted = a; passer = b
            else:
                accepted = b; passer = a
            # If freight and passenger conflict, force freight to be accepter only if required? per user: freight yields always
            # so if accepted is freight and passer is passenger -> invert (passenger should be accepted)
            if priority_map[accepted['type']] < priority_map[passer['type']]:
                # swap: make passenger accepted
                accepted, passer = passer, accepted
            # compute accepted_dep (if no explicit dep, approximate by arr + dwell)
            # find per_train_stops info to get station stop index for accepted
            # we need stop_seq for accepted at the station either left or right
            # find station name near block: use left station for A->B accepted from left etc.
            # Simplify: extract accepted train's stop record that has station matching block border:
            # We'll search its stops for station equal to station_left or station_right and get dep
            acc_stops = per_train_stops[accepted['train_id']]
            pass_stops = per_train_stops[passer['train_id']]
            # find station where overlap occurs: either station_left or station_right
            # for simplicity, choose the station whose event time is closest to accepted['entry']
            # find station index in accepted stops with arr near accepted.entry
            def find_closest_stop(stops, time):
                best = None
                best_diff = 1e9
                for idx, s in enumerate(stops):
                    t0 = s['arr']
                    diff = abs(t0 - time)
                    if diff < best_diff:
                        best = (idx, s)
                        best_diff = diff
                return best
            acc_idx, acc_stop = find_closest_stop(acc_stops, accepted['entry'])
            pass_idx, pass_stop = find_closest_stop(pass_stops, passer['entry'])
            # accepted_dep:
            acc_dep = acc_stop['dep'] if acc_stop['dep'] is not None else acc_stop['arr'] + acc_stop['dwell']
            min_pass_time = acc_dep + CLEARANCE
            # passer's current entry time:
            pass_entry = pass_stop['arr']
            # if passer too early, delay passer so that its entry >= min_pass_time
            if pass_entry < min_pass_time:
                delay_needed = min_pass_time - pass_entry
                delays[passer['train_id']] += delay_needed
            # after passer clears, accepted must wait WAIT_AFTER_PASS before departing
            # passer_clear_time estimate:
            # use passer's dep if exists else arr + small epsilon -> but we will recompute in next iteration
            # For simplicity, force accepted to be delayed so its next movement occurs not earlier than passer_clear + WAIT_AFTER_PASS
            # compute passer_clear as pass_entry + (very small pass time) -> but safer to use pass_stop['dep'] if exists
            passer_dep = pass_stop['dep'] if pass_stop['dep'] is not None else pass_stop['arr']
            min_acc_dep = passer_dep + WAIT_AFTER_PASS
            if acc_dep < min_acc_dep:
                # delay accepted's subsequent stops
                # amount to add to accepted's schedule from acc_idx+1 onwards:
                add = min_acc_dep - acc_dep
                delays[accepted['train_id']] += add
            # continue loop (we applied delays to delays dict; next iteration will rebuild intervals)
        elif c['type'] == 'opposite_meet_nosiding':
            # no siding: block overlap: delay lower-priority train until block free
            # decide which to delay: lower priority or later entry
            if pri_a < pri_b:
                to_delay = a
                anchor = b
            elif pri_b < pri_a:
                to_delay = b
                anchor = a
            else:
                # equal priority: delay later entry one
                if a['entry'] <= b['entry']:
                    to_delay = b; anchor = a
                else:
                    to_delay = a; anchor = b
            # delay to_delay so its entry >= anchor.release + small epsilon (1 min)
            needed = anchor['release'] + 1.0 - to_delay['entry']
            if needed < 0:
                needed = 0.0
            delays[to_delay['train_id']] += needed
        elif c['type'].startswith('avto') or c['type'].startswith('yarim'):
            # same-direction conflict: we delay the later starting train to satisfy AVTO_I or YARIM_WAIT
            a_entry = a['entry']; b_entry = b['entry']
            if a_entry <= b_entry:
                later = b; earlier = a
            else:
                later = a; earlier = b
            if c['type'] == 'avto_same_dir_I':
                needed = earlier['entry'] + AVTO_I - later['entry'] + 0.1
                if needed < 0: needed = 0.0
                delays[later['train_id']] += needed
            else:
                # yarim_same_dir_wait
                needed = earlier['release'] + YARIM_WAIT - later['entry'] + 0.1
                if needed < 0: needed = 0.0
                delays[later['train_id']] += needed
        else:
            # fallback: delay lower priority
            if pri_a <= pri_b:
                to_delay = a; anchor = b
            else:
                to_delay = b; anchor = a
            needed = anchor['release'] + 1.0 - to_delay['entry']
            if needed < 0: needed = 0.0
            delays[to_delay['train_id']] += needed
        # rebuild intervals will happen at top of while loop
    # return final delays and final schedule
    final_trains = [dict(tr, start_time=min_to_hhmm(hhmm_to_min(tr['start_time']) + delays[tr['id']])) for tr in trains]
    intervals_final, per_train_stops_final = build_intervals_from_schedule(final_trains)
    return delays, intervals_final, per_train_stops_final

# Pretty-print conflicts to CSV
def export_conflicts(intervals_df, per_train_stops, fname='conflicts_report.csv'):
    confs = detect_conflicts(intervals_df, per_train_stops)
    rows = []
    for c in confs:
        # flatten a and b
        a = c['a']; b = c['b']
        rows.append({
            'block': c['block'],
            'type': c['type'],
            'a.train': a['train_id'], 'a.type': a['type'], 'a.dir': a['dir'], 'a.entry_min':a['entry'], 'a.release_min':a['release'],
            'b.train': b['train_id'], 'b.type': b['type'], 'b.dir': b['dir'], 'b.entry_min':b['entry'], 'b.release_min':b['release']
        })
    dfc = pd.DataFrame(rows)
    dfc.to_csv(fname, index=False)
    print(f"Saved conflicts to {fname}")

# Plot time-distance diagram
def plot_time_distance(per_train_stops, fname='time_distance.png'):
    plt.figure(figsize=(12,6))
    # y axis: station positions (index)
    station_index = {s:i for i,s in enumerate(stations)}
    for tid, stops in per_train_stops.items():
        xs = []
        ys = []
        for s in stops:
            # use arr time for point if available else dep
            t = s['arr'] if s['arr'] is not None else s['dep']
            if t is None: continue
            xs.append(t/60.0)  # hours
            ys.append(station_index[s['station']])
            # show dwell as small vertical marker by duplicating time at same station with dep
            if s['dep'] is not None and s['dep'] != s['arr']:
                xs.append(s['dep']/60.0)
                ys.append(station_index[s['station']])
        if len(xs) >= 2:
            plt.plot(xs, ys, marker='o')  # do not set colors explicitly per rules
            plt.text(xs[0], ys[0], tid)
    plt.yticks(range(len(stations)), stations)
    plt.xlabel("Time (hours)")
    plt.title("Time-Distance diagram (station index on Y)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.show()
    print(f"Saved time-distance diagram to {fname}")

# -------------------------
# MAIN FLOW
# -------------------------
if __name__ == "__main__":
    print("Building initial schedule...")
    intervals_df, per_train_stops = build_intervals_from_schedule(trains)
    print("Detecting initial conflicts...")
    conflicts = detect_conflicts(intervals_df, per_train_stops)
    print(f"Initial conflicts found: {len(conflicts)}")
    # Export initial conflicts for inspection
    export_conflicts(intervals_df, per_train_stops, fname='conflicts_initial.csv')
    print("Resolving conflicts with greedy resolver (respecting priorities and passing rules)...")
    delays, intervals_final, per_train_stops_final = greedy_resolve(trains, intervals_df, per_train_stops)
    print("Delays applied (minutes):", delays)
    # final conflicts after resolution
    final_conflicts = detect_conflicts(intervals_final, per_train_stops_final)
    print(f"Conflicts after resolution: {len(final_conflicts)}")
    export_conflicts(intervals_final, per_train_stops_final, fname='conflicts_report.csv')
    plot_time_distance({tid: stops for tid, stops in per_train_stops_final.items()}, fname='time_distance.png')
    # Save final schedule to CSV
    rows = []
    for tid, stops in per_train_stops_final.items():
        for s in stops:
            rows.append({'train_id': tid, 'station': s['station'], 'arr_min': s['arr'], 'dep_min': s['dep'], 'dwell': s['dwell']})
    pd.DataFrame(rows).to_csv('final_schedule.csv', index=False)
    print("Saved final_schedule.csv")
    print("Done.")
