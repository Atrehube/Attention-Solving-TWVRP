import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from generate_data import generate_vrp_data
from utils import load_model
from problems import CVRP, TWCVRP
from matplotlib import pyplot as plt
from or_tools import or_tools_twvrp
from options import get_options

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def discrete_cmap(N, base_cmap=None):
  """
    Create an N-bin discrete colormap from the specified input map
    """
  # Note that if base_cmap is a string or None, you can simply do
  #    return plt.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)

def plot_vehicle_routes(data, route, ax1, markersize=5, visualize_demands=False, visual_demand_legend = True, demand_scale=1, round_demand=False,from_AM = True, time = None):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    
    # route is one sequence, separating different routes with 0 (depot)
    if from_AM:
        routes = [r[r!=0] for r in np.split(route.cpu().numpy(), np.where(route==0)[0]) if (r != 0).any()]
    else:
        routes = [r for r in route if len(r) > 2]
        
    depot = data['depot'].cpu().numpy()
    locs = data['loc'].cpu().numpy()
    demands = data['demand'].cpu().numpy() * demand_scale
    capacity = demand_scale # Capacity is always 1
    
    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize*4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    legend = ax1.legend(loc='upper center')
    
    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        r= np.array(r)
        #print(r,type(r))
        color = cmap(len(routes) - veh_number) # Invert to have in rainbow order
        
        if not from_AM:
            r = np.delete(r, np.where(r == 0))

        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        #if from_AM:
        route_demands = np.insert(route_demands,0, 0)
        route_demands = np.append(route_demands,[0],)
        coords = np.insert(coords, 0, depot,axis = 0)
        coords = np.append(coords,[depot],axis = 0)
        xs, ys = coords.transpose()

        total_route_demand = sum(route_demands)
        #assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
        
        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
            
            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))
            
            x_prev, y_prev = x, y
            cum_demand += d
            
        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='R{}, # {}, c {} / {}, d {:.2f}'.format(
                veh_number, 
                len(r), 
                int(total_route_demand) if round_demand else total_route_demand, 
                int(capacity) if round_demand else capacity,
                dist
            ) if visual_demand_legend else 
            'R{}, # {}, d {:.2f}'.format(
                veh_number, 
                len(r), 
                dist
            )
        )
        
        qvs.append(qv)
    if time is None:
        ax1.set_title('{} route(s), total distance {:.2f}'.format(len(routes), total_dist))
    else:
        if from_AM:
            ax1.set_title('AM: {} route(s), total distance {:.2f}, total time {:.2f}'.format(len(routes), total_dist, time))
        else:
            ax1.set_title('OR-Tools: {} route(s), total distance {:.2f}, total time {:.2f}'.format(len(routes), total_dist, time))
   
    ax1.legend(handles=qvs)
    
    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')
    
    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)
        
def construct_comparison_graph(model_path,problem):
    model, _ = load_model(model_path)
    torch.manual_seed(1234)
    dataset = CVRP.make_dataset(size=20, num_samples=1)
    # Need a dataloader to batch instances
    dataloader = DataLoader(dataset, batch_size=1000)
    
    visual_demand_legend = True if problem == 'twcvrp' else False
    
    # Make var works for dicts
    batch = next(iter(dataloader))
    
    # Run the model
    model.eval()
    model.set_decode_type('greedy')
    with torch.no_grad():
        length, log_p, pi = model(batch, return_pi=True)
    times = length
    tours = pi
    for i, (data, tour, time) in enumerate(zip(dataset, tours,times)):
        fig, axs = plt.subplots(1,2, figsize=(16,8))
        fig.suptitle(problem.upper())
        plot_vehicle_routes(data, tour, axs[0], visualize_demands=False, visual_demand_legend = visual_demand_legend, demand_scale=100, round_demand=True, from_AM= True,time = time)
        opts = get_options(['--problem',problem,'--eval_only'])
        cost_or, pi_or = or_tools_twvrp(dataset, opts,True)
        cost_or = cost_or/10_000 # or_tools uses discrete time, so it is scaled up. 
        plot_vehicle_routes(data, pi_or, axs[1], visualize_demands=False, visual_demand_legend = visual_demand_legend, demand_scale=100, round_demand=True, from_AM = False,time = cost_or)
        
if __name__ == '__main__':
    construct_comparison_graph('outputs/twcvrp_20/run_20210422T021640', 'twcvrp')
    construct_comparison_graph('outputs/twvrp_20/run_20210421T211407', 'twvrp')      
