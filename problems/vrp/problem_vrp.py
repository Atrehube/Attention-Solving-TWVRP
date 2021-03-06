from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.vrp.state_cvrp import StateCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from problems.vrp.state_vrp import StateVRP
from problems.vrp.state_twvrp import StateTWVRP
from problems.vrp.state_twcvrp import StateTWCVRP
from utils.beam_search import beam_search

class VRP(object):
    NAME = 'vrp' # Vehicle Routing Problem
    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"
        
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None
        
    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs) # Could define a new dataset method w/o demands/caps to improve efficiency.

    @staticmethod
    def make_state(*args, **kwargs):
        return StateVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = VRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TWVRP(object):
    NAME = 'twvrp' # Vehicle Routing Problem
    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"


        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        # collect the number of missed locations per trial
        
        # Reconstruct waiting times.
        
        times_with_depot = torch.cat(
            (
            torch.full_like(dataset['start_time'][:,:1], 0),
            dataset['start_time']
            ),
            1
        )
        
        # Get the start times, arange according to policy
        start_times = times_with_depot.gather(1,pi)

       
        # Find the travel times in distance
        travel_times = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2)
        # Add a 0 to the end of travel time so the dimensions match
        travel_times = torch.cat((travel_times, torch.zeros(travel_times.shape[0],1,device = travel_times.device)),dim =1)
        travel_times_master = torch.clone(travel_times)
        
        for step in range(pi.shape[1]):
            # each cost is the maximum of the previous cost + change , or time window.
            # get previous travel time
            prev = torch.cat((torch.zeros((travel_times.shape[0],1),device = travel_times.device),travel_times[:,:-1]),dim = 1)
            # current step is the maximum of ( previous cost + travel_time, or time window bound )
            travel_times[:,step] = torch.max(start_times[:,step],travel_times[:,step] + prev[:,step])
            # Reset the cum sum at each depot. 
            travel_times = torch.where(pi == 0, travel_times_master,travel_times)
        
        # Get the marginal change in travel time. Mimicks original travel_times but now has jumps for waiting. 
        marginal = (travel_times[:, 1:] - travel_times[:, :-1])
        marginal = torch.cat((torch.zeros((marginal.shape[0],1),device = travel_times.device),marginal),dim = 1)
        # When returning to the depot, overwrite with orginal distance to return to depot. 
        marginal = torch.where(pi == 0, travel_times_master,marginal)
                          
        return (
            marginal.sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last 
        ), None
    
        
    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs) # Could define a new dataset method w/o demands/caps to improve efficiency.

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTWVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TWVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

class TWCVRP(object):

    NAME = 'twcvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"
        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"



        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))


            # Reconstruct waiting times.
        
        times_with_depot = torch.cat(
            (
            torch.full_like(dataset['start_time'][:,:1], 0),
            dataset['start_time']
            ),
            1
        )
        
        # Get the start times, arange according to policy
        #start_times = times_with_depot.gather(1,pi)
        
        #for _ in range(pi.shape[1]):
        #    forward_fill = torch.cat((torch.zeros((start_times.shape[0],1),device = start_times.device),start_times[:,:-1]),dim = 1)
        #    start_times = torch.where(torch.logical_and(start_times < forward_fill, start_times != 0), forward_fill, start_times)

        # Find the extra waiting times
        #excess = torch.where(start_times != 0 , (start_times - torch.cat((torch.zeros((start_times.shape[0],1),device = start_times.device),start_times[:,:-1]),dim = 1)),start_times)

        
        # Get the start times, arange according to policy
        start_times = times_with_depot.gather(1,pi)

        travel_times = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2)
        # Add a 0 to the end of travel time so the dimensions match
        travel_times = torch.cat((travel_times, torch.zeros(travel_times.shape[0],1,device = travel_times.device)),dim =1)
        travel_times_master = torch.clone(travel_times)
        
        for step in range(pi.shape[1]):
            # each cost is the maximum of the previous cost + change , or time window.
            # get previous travel time
            prev = torch.cat((torch.zeros((travel_times.shape[0],1),device = travel_times.device),travel_times[:,:-1]),dim = 1)
            # current step is the maximum of ( previous cost + travel_time, or time window bound )
            travel_times[:,step] = torch.max(start_times[:,step],travel_times[:,step] + prev[:,step])
            # Reset the cum sum at each depot. 
            travel_times = torch.where(pi == 0, travel_times_master,travel_times)
        
        # Get the marginal change in travel time. Mimicks original travel_times but now has jumps for waiting. 
        marginal = (travel_times[:, 1:] - travel_times[:, :-1])
        marginal = torch.cat((torch.zeros((marginal.shape[0],1),device = travel_times.device),marginal),dim = 1)
        # When returning to the depot, overwrite with orginal distance to return to depot. 
        marginal = torch.where(pi == 0, travel_times_master,marginal)


        # Get the start times, arange according to policy
        #start_times = times_with_depot.gather(1,pi)
        # Find the travel times in distance
        #travel_times = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2)
        # Add a 0 to the end of travel time so the dimensions match
        #travel_times = torch.cat((travel_times, torch.zeros(travel_times.shape[0],1,device = travel_times.device)),dim =1)
        # Create a tour length that resets to 0 when the depot is visitied again
        #partials = torch.where(pi == 0, torch.zeros(*pi.size(),device = travel_times.device),travel_times)
        # Find the cumulative value of this route.
        #cumulative = partials.cumsum(dim = 1)
        # Find where the routes, restart
        #restart = torch.where(partials != 0,torch.zeros(*pi.size(),device = travel_times.device),cumulative)
        # Propagate the 'reset' value forward. This is the trip length at the first stop.
        # Loop through the lenght of policy to ensure that propagation completes.
        #for _ in range(pi.shape[1]):
        #    forward_fill = torch.cat((torch.zeros((restart.shape[0],1),device = restart.device),restart[:,:-1]),dim = 1)
        #    restart = torch.where(restart == 0, restart + forward_fill, restart)
        # The result is a cumulative length of each trip.
        #results = cumulative - restart
            
        # The excess waiting time is equal to the maximum discrepancy in travel length compared to opening time.
        #excess_waiting = torch.where(start_times - results > 0, start_times - results, torch.zeros(*pi.size(),device = pi.device)).max(dim = 1)[0]
                                  
        return (
             marginal.sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last 
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTWCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TWCVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                    'start_time': (torch.FloatTensor(size).uniform_(0,5)).float(),
                    'end_time': torch.FloatTensor(size).uniform_(6,10).float()
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
