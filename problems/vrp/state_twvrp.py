import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateTWVRP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    coords: torch.Tensor  # Depot + loc
    start_time: torch.Tensor
    end_time: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    time: torch.Tensor # For time windows. Keep track of time.
    
    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            time= self.time[key],
            cur_coord=self.cur_coord[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        depot = input['depot']
        loc = input['loc']
        start_time = input['start_time']
        end_time = input['end_time']

        batch_size, n_loc, _ = loc.size()
        return StateTWVRP(
            loc = loc,
            coords=torch.cat((depot[:, None, :], loc), -2),
            start_time = start_time,
            end_time = end_time,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            time=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        distance = (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        wait_time = torch.gather(torch.cat((torch.zeros(distance.shape[0],1,device = distance.device),self.start_time),dim = 1), 1,selected) - self.lengths
        wait_time = torch.where(wait_time >= 0, wait_time, torch.zeros(wait_time.shape,device = distance.device))
        
        lengths = self.lengths + distance + wait_time 
        time = (self.time + distance + wait_time)*(prev_a != 0).float() 
        
        
        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1, time = time
        )

    def all_finished(self):
        # Finished if time is ge than number of locations, and (all places visited, or current time is past time window)
        return self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited locations.
        0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.loc.size(-1))

        distance = (self.coords - self.cur_coord).norm(p=2, dim=-1)
        time_constraint = (self.end_time[self.ids,:] < (self.time[:,:,None].repeat(1,1,visited_loc.shape[2]) + distance[:,1:].reshape(visited_loc.shape[0],1,visited_loc.shape[2])))
        # Nodes that cannot be visited are already visited or too much demand to be served now
        
        mask_loc = visited_loc  | time_constraint

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        #exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        # Nodes that cannot be visited are already visited or too much demand to be served now

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)

    def construct_solutions(self, actions):
        return actions
# -*- coding: utf-8 -*-

