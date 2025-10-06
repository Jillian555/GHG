import torch.nn.functional as F
from .gnns import GAT, GCN, GIN, SGC
def get_model(args):
    if args.backbone == 'GAT':
        heads = ([args.GAT_args['heads']] * args.GAT_args['num_layers']) + [args.GAT_args['out_heads']]
        model = GAT(args, heads, F.elu)
    elif args.backbone == 'GCN':
        model = GCN(args)
    elif args.backbone == 'GIN':
        model = GIN(args)
    elif args.backbone == 'SGC':
        model = SGC(args)
    return model

