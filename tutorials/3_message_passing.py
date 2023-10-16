import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


# although DGL has builtin support of GraphSAGE via dgl.nn.SAGEConv
# here is how you can implement GraphSAGE convolution in DGL by your own
class SAGEConv(nn.Module):
    '''Graph convolution module used by the GraphSAGE model.

    Parameters
    -----------
    in_feat: int
        Input feature size.
    out_feat: int
        Output feature size.
    '''
    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        '''Forward computation

        Parameters
        ----------
        g: Graph
            The input graph
        h: Tensor
            The input node feature
        '''
        with g.local_scope():
            g.ndata['h'] = h
            # update_all is a message passing API
            g.update_all(
                message_func = fn.copy_u('h', 'm'),
                reduce_func = fn.mean('m', 'h_N'),
            )
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)
    
    
