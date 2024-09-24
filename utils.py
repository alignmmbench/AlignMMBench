def split_data_for_distributed_nodes(args, data_length):
    """Split dataset for distributed nodes

    Args:
        args (argparse.Namespace): rank, world_size, model_parallel_size
        data_length (int): length of data
    """
    rank = args.rank
    world_size = args.world_size
    mp_size = args.model_parallel_size
    rank_group_size = world_size // mp_size
    # add padding data
    if data_length % rank_group_size == 0:
        pad_data_len = 0
    else:
        pad_data_len = (data_length // rank_group_size + 1) * rank_group_size - data_length
    # split data
    distributed_data_indices = []
    for i in range(data_length + pad_data_len):
        if i % rank_group_size == rank // mp_size:
            distributed_data_indices.append(i % data_length)
    return distributed_data_indices
