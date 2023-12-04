import torch
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import numpy as np
from importlib.machinery import SourceFileLoader
import random as rd

path = './utils.py'
utils = SourceFileLoader('utils', path).load_module()

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    ids = [item[2] for item in batch]
    return [data, target], ids


class FlowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 list_IDs: List[str],
                 tileid2oa2features2vals: Dict,
                 o2d2flow: Dict,
                 oa2features: Dict,
                 oa2pop: Dict,
                 oa2centroid: Dict,
                 dim_dests: int,
                 frac_true_dest: float, 
                 model: str,
                 mode: str,
                 dict_od: Dict
                ) -> None:
        'Initialization'
        self.list_IDs = list_IDs
        self.tileid2oa2features2vals = tileid2oa2features2vals
        self.o2d2flow = o2d2flow
        self.oa2features = oa2features
        self.oa2pop = oa2pop
        self.oa2centroid = oa2centroid
        self.dim_dests = dim_dests
        self.frac_true_dest = frac_true_dest
        self.model = model
        self.oa2tile = {oa:tile for tile,oa2v in tileid2oa2features2vals.items() for oa in oa2v.keys()}
        self.mode = mode
        self.dict_od = dict_od
        self.origin_fakes={}

    def __len__(self) -> int:
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def get_features(self, oa_origin, oa_destination):
        oa2features = self.oa2features
        oa2centroid = self.oa2centroid
        dist_od = utils.earth_distance(oa2centroid[oa_origin], oa2centroid[oa_destination])


        return oa2features[oa_origin] + oa2features[oa_destination] + [dist_od]

    def get_flow(self, oa_origin, oa_destination):
        o2d2flow = self.o2d2flow
        try:
            if self.mode=='test' and  oa_destination in self.dict_od[oa_origin]:
                if o2d2flow[oa_origin][oa_destination]== 0 or o2d2flow[oa_origin][oa_destination]==0.0:
                    print('test is 0!')
                return o2d2flow[oa_origin][oa_destination]
            if self.mode=='test' and  oa_destination not in self.dict_od[oa_origin]:
                return 0
            elif self.mode=='train':
                return o2d2flow[oa_origin][oa_destination]
            else:
                return 0
        except KeyError:
            if self.mode=='test':# and  oa_destination in self.dict_od[oa_origin]:
                print('Test flow not found',oa_origin,oa_destination)
            return 0

    def get_destinations(self, oa, size_train_dest, all_locs_in_train_region):
        o2d2flow = self.o2d2flow
        frac_true_dest = self.frac_true_dest
        try:
            true_dests_all = list(o2d2flow[oa].keys())
        except KeyError:
            true_dests_all = []
           
        if self.mode=='test':
            dests = true_dests_all
           
            for dtest in self.dict_od[oa]:
                if dtest not in dests:
                    print(self.mode,'lost dest')
        else:
           
            size_true_dests = min(int(size_train_dest * frac_true_dest), len(true_dests_all))
            size_fake_dests = size_train_dest - size_true_dests

            true_dests = np.random.choice(true_dests_all, size=size_true_dests, replace=False)
            fake_dests_all = list(set(all_locs_in_train_region) - set(true_dests))
            fake_dests = np.random.choice(fake_dests_all, size=size_fake_dests, replace=False)
            self.origin_fakes[oa]=fake_dests

            dests = np.concatenate((true_dests, fake_dests))
            np.random.shuffle(dests)
           
            #print('Deep Gravity fake destinations',size_true_dests,size_fake_dests,len(true_dests_all),size_train_dest - size_true_dests)

        return dests
    
    def get_X_T(self, origin_locs, dest_locs):

        X, T = [], []
        for en, i in enumerate(origin_locs):
            X += [[]]
            T += [[]]
            for j in dest_locs[en]:
                X[-1] += [self.get_features(i, j)]
                T[-1] += [self.get_flow(i, j)]

        teX = torch.from_numpy(np.array(X)).float()
        teT = torch.from_numpy(np.array(T)).float()
        #print(self.mode, len([1 for it in teT if it>0.]))
        return teX, teT

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        tileid2oa2features2vals = self.tileid2oa2features2vals
        dim_dests = self.dim_dests
        oa2tile = self.oa2tile

        # Select sample (tile)
        sampled_origins = [self.list_IDs[index]]

        all_locs_in_train_region = self.list_IDs
        size_train_dest = min(dim_dests, len(set(all_locs_in_train_region)))
        #print('getitem',self.mode,len(sampled_origins),len(all_locs_in_train_region),size_train_dest)
        sampled_dests = [self.get_destinations(oa, size_train_dest, all_locs_in_train_region)
                         for oa in sampled_origins]
            
        sampled_trX, sampled_trT = self.get_X_T(sampled_origins, sampled_dests)

#         print(len(all_locs_in_train_region))
        return sampled_trX, sampled_trT, sampled_origins

    def __getitem_tile__(self, index: int) -> Tuple[Any, Any]:
        'Generates one sample of data (one tile)'
    
        tileid2oa2features2vals = self.tileid2oa2features2vals
        dim_dests = self.dim_dests
        tile_ID = self.list_IDs[index]
        sampled_origins = list(tileid2oa2features2vals[tile_ID].keys())

        # Select a subset of OD pairs
        train_locs = sampled_origins
        all_locs_in_train_region = train_locs
        size_train_dest = min(dim_dests, len(all_locs_in_train_region),)
        #print('getitemtile',self.mode,len(sampled_origins),len(all_locs_in_train_region),size_train_dest)
        sampled_dests = [self.get_destinations(oa, size_train_dest, all_locs_in_train_region)
                         for oa in sampled_origins]

        # get the features and flows
        sampled_trX, sampled_trT = self.get_X_T(sampled_origins, sampled_dests)

        return sampled_trX, sampled_trT

