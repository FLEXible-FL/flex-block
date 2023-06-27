from flexBlock.pool.decorators import concensus


@concensus(deterministic=False)
def proof_of_work(aggregators, blockchain, *args, **kwargs):
    return 1
