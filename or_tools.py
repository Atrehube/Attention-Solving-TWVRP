# File to convert problem into format that can be solved by OR-TOOLS.
# Solve problems in batch, then report solution times and speed of calculation.  

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from datetime import datetime

def or_tools_twvrp(dataset,opts,return_pi = False):
    problems = dataset.data
    costs = []
    if opts.problem == 'twvrp':
        capacitated = False
    elif opts.problem == 'twcvrp':
        capacitated = True
    else:
        raise Exception('Only TWVRP and TWCVRP have OR-TOOLS implementation')

    for problem in problems:
        data = create_data_model(problem, capacitated)
        
        start_time = datetime.now()
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], data['depot'])
        
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)
        
        # Create and register a transit callback.
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        
        #if capacitated:
        if capacitated:
            # Add Capacity constraint.
            def demand_callback(from_index):
                """Returns the demand of the node."""
                # Convert from routing variable Index to demands NodeIndex.
                from_node = manager.IndexToNode(from_index)
                return data['demand'][from_node]
        
            demand_callback_index = routing.RegisterUnaryTransitCallback(
                demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # null capacity slack
                data['vehicle_capacities'],  # vehicle maximum capacities
                True,  # start cumul to zero
                'Capacity')
        
        
        # Add Time Windows constraint.
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            5000000,  # allow waiting time
            1000000,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time)
        time_dimension = routing.GetDimensionOrDie(time)

        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == data['depot']:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(round(time_window[0]), round(time_window[1]))
        # Add time window constraints for each vehicle start node.
        depot_idx = data['depot']
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data['time_windows'][depot_idx][0],
                data['time_windows'][depot_idx][1])
            

    
        # Instantiate route start and end times to produce feasible times.
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i)))
            # Setting first solution heuristic.

        


        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
    
        # Print solution on console.
        if solution:
            total_time = 0
            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)
                while not routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    index = solution.Value(routing.NextVar(index))
                time_var = time_dimension.CumulVar(index)
                total_time += solution.Min(time_var)
        costs.append(total_time/10000)
    if return_pi:
        time_dimension = routing.GetDimensionOrDie('Time')
        total_time = 0
        pi = []
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = []
            while not routing.IsEnd(index):
                #time_var = time_dimension.CumulVar(index)
                plan_output.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            time_var = time_dimension.CumulVar(index)
            plan_output.append(manager.IndexToNode(index))

            total_time += solution.Min(time_var)
            pi.append(plan_output)
        return total_time, pi
    else:
        return costs
        
def create_data_model(problem,has_capacity):
    data ={}
    
    points = np.array(problem['loc'])
    points = np.insert(points,0,problem['depot'],axis = 0)
    data['points'] = points
    data['points'] = [[int(round(x*10000)), int(round(y*10000))] for x,y in data['points']]
    x,y = points.T
    z = np.array([complex(p[0], p[1]) for p in points])
    m, n = np.meshgrid(z, z)
    b = abs(m-n) 
    
    # Find distance between points
    b_symm = (b + b.T)/2
    np.fill_diagonal(b_symm,0)
    data['time_matrix'] = b_symm.tolist()
    
    data['time_windows'] = list(zip(problem['start_time'].numpy(), problem['end_time'].numpy()))
    data['time_windows'] = np.insert(data['time_windows'],0,[0,10000],axis = 0).tolist()
    data['num_vehicles'] = len(data['time_windows'])
    data['depot'] = 0
    if has_capacity:
        data['demand'] = problem['demand'].numpy()
        data['demand'] = np.insert(data['demand'],0,0,axis = 0).tolist()
        
    # Multiply everything by 100 then round, since OR-TOOLS doesn't play nice with floats. 
        data['demand'] = [int(round(x * 10000)) for x in data['demand']]
        data['vehicle_capacities'] = [10000 for x in range(data['num_vehicles'])]

    data['time_windows'] = [[int(round(x*10000)), int(round(y*10000))] for x,y in data['time_windows']]
    data['time_matrix'] = (np.array(data['time_matrix'])*10000).round().astype(int).tolist()
    
    
    return data
        
def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), solution.Min(time_var),
                solution.Max(time_var))
            index = solution.Value(routing.NextVar(index))
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                    solution.Min(time_var),
                                                    solution.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            solution.Min(time_var))
        print(plan_output)
        total_time += solution.Min(time_var)
    print('Total time of all routes: {}min'.format(total_time/10000))
