import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import random
import gymnax_exchange.jaxob.JaxOrderBookArrays as job
"""hamilton_apportionment_permuted_jax: A utility function using JAX, 
                                     implementing a Hamilton apportionment 
                                     method with randomized seat allocation."""


def argsort_rev(arr):
    """ 'arr' sorted in descending order (LTR priority tie-breaker) """
    return (arr.shape[0] - 1 - jnp.argsort(arr[::-1]))[::-1]

def rank_rev(arr):
    """ Rank array in descending order, with ties having left-to-right priority. """
    return jnp.argsort(argsort_rev(arr))

@jax.jit
def clip_by_sum_int(a: jax.Array, max_sum: int) -> jax.Array:
    """ Clip a vector so that its sum is at most max_sum as an integer,
        while preserving the relative proportions of the elements.
        Ties have left-to-right priority.

        ex: clip_by_sum_int(jnp.array([3, 2, 3, 1]), 8)) -->  [3 2 2 1]

    Args:
        a: The vector to clip.
        max_sum: The maximum sum of the vector.

    Returns:
        The clipped vector.
    """
    def clip(a: jax.Array, a_sum: int) -> jax.Array:
        a, remainders = jnp.divmod(a * max_sum, a_sum)
        rest = max_sum - jnp.sum(a)
        ranks = rank_rev(remainders)
        
        # add 1 to first 'rest' elements of original 'a' with highest remainder
        a = jnp.where(
            ranks < rest,
            a + 1,
            a,
        )
        return a

    a_sum = jnp.sum(a)
    return jax.lax.cond(
        a_sum > max_sum,
        lambda: clip(a, a_sum),
        lambda: a,
    )
from functools import partial
@partial(jax.vmap, in_axes=(0, None))
def p_in_cnl(p, prices_cnl):
    return jnp.where((prices_cnl == p) & (p != 0), True, False)
def matching_masks(prices_a, prices_cnl):
    res = p_in_cnl(prices_a, prices_cnl)
    return jnp.any(res, axis=1), jnp.any(res, axis=0)


def tree_stack(trees):
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)

def tree_unstack(tree):
    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]

def array_index(array,index):
    return array[index]

@jax.jit
def index_tree(tree,index):
    array_index = lambda array,index : array[index]
    indeces=[index]*len(jtu.tree_flatten(tree)[0])
    tree_indeces=jtu.tree_unflatten(jtu.tree_flatten(tree)[1],indeces)
    return jtu.tree_map(array_index,tree,tree_indeces)

def hamilton_apportionment_permuted_jax(votes, seats, key):
    """
    Compute the Hamilton apportionment method with permutation using JAX.

    Args:
        votes (jax.Array): Array of votes for each party/entity.
        seats (int): Total number of seats to be apportioned.
        key (chex.PRNGKey): JAX key for random number generation.

    Returns:
        jax.Array: Array of allocated seats to each party/entity.
    """
    std_divisor = jnp.sum(votes) / seats # Calculate the standard divisor.
    # Initial allocation of seats based on the standard divisor and compute remainders.
    init_seats, remainders = jnp.divmod(votes, std_divisor)
    # Compute the number of remaining seats to be allocated.
    remaining_seats = jnp.array(seats - init_seats.sum(), dtype=jnp.int32) 
    # Define the scanning function for iterative seat allocation.
    def allocate_remaining_seats(carry,x): # only iterate 4 times, as remaining_seats in {0,1,2,3}
        key,init_seats,remainders = carry
        key, subkey = jax.random.split(key)
        # Create a probability distribution based on the maximum remainder.
        distribution = (remainders == remainders.max())/(remainders == remainders.max()).sum()
        # Randomly choose a party/entity to allocate a seat based on the distribution.
        chosen_index = jax.random.choice(subkey, remainders.size, p=distribution)
        # Update the initial seats and remainders for the chosen party/entity.
        updated_init_seats = init_seats.at[chosen_index].add(jnp.where(x < remaining_seats, 1, 0))
        updated_remainders = remainders.at[chosen_index].set(0)
        return (key, updated_init_seats, updated_remainders), x
        # Iterate over parties/entities to allocate the remaining seats.
    (key, init_seats, remainders), _ = jax.lax.scan(
                                                    allocate_remaining_seats,
                                                    (key, init_seats, remainders), 
                                                    xs=jnp.arange(votes.shape[0])
                                                    )
    return init_seats


def create_init_book(cfg:job.Configuration,
                     order_capacity=10,
                     trade_capacity=10,
                     pricerange=[2190000,2200000,2210000],
                     quantrange=[0,500],
                     timeinit=[34200,0],
                     percent_fill=0.9):
    """
    Generates a random orderbook state for a given maximum capacity for orders and trades. 
    Random prices/quantities generated by uniform sampling in pricerange/quantrange. 
    """
    qtofill = int(order_capacity*percent_fill) #fill available space.
    asks=[]
    bids=[]
    orderid=cfg.init_id
    traderid=cfg.init_id
    times=timeinit[0]
    timens=timeinit[1]
    for i in range(qtofill):
        asks.append([random.randint(pricerange[1],pricerange[2]),random.randint(quantrange[0],quantrange[1]),orderid,traderid,times,timens])
        bids.append([random.randint(pricerange[0],pricerange[1]),random.randint(quantrange[0],quantrange[1]),orderid-1,traderid-1,times,timens])
        orderid-=2
        traderid-=2
    bids=jnp.concatenate((jnp.array(bids),
                          jnp.ones((order_capacity-qtofill,job.cst.ORDERBOOK_FEAT),dtype=jnp.int32)*job.cst.EMPTY_SLOT),
                          axis=0)
    asks=jnp.concatenate((jnp.array(asks),
                          jnp.ones((order_capacity-qtofill,job.cst.ORDERBOOK_FEAT),dtype=jnp.int32)*job.cst.EMPTY_SLOT),
                          axis=0)
    trades=jnp.ones((trade_capacity,job.cst.TRADE_FEAT),dtype=jnp.int32)*job.cst.EMPTY_SLOT
    return asks,bids,trades

def create_rand_message(type='limit',
                        side=None,
                        price_range=[2100000,2200000],
                        quant_range=[0,500],
                        prev_time=job.cst.STARTOFDAY,
                        times_range=[0,2],
                        timens_range=[0,job.cst.NS_PER_SEC]):

    type_options = ['limit', 'cancel', 'market']
    side_options = ['bid', 'ask']
    if type == None:
        type = type = random.choice(type_options)
    if side == None:
        side = random.choice(side_options)
    price = random.randint(price_range[0], price_range[1])
    quant = random.randint(quant_range[0], quant_range[1])
    delta_times = random.randint(times_range[0], times_range[1])
    delta_timens = random.randint(timens_range[0], timens_range[1])
    
    
    return create_message(type=type,
                            side=side,
                            price=price,
                            quant=quant,
                            times=prev_time[0]+delta_times,
                            timens=prev_time[1]+delta_timens)

def create_message(type='limit',side='bid',price=2200000,quant=10,times=36000,timens=0,id=8888):
    """
    Generates a specific message (based on human-editable inputs)
    Outputs both the 'dictionary' format 
    """
    if type=='limit':
        type_num=job.cst.MessageType.LIMIT.value
    elif type =='cancel' or type == 'delete':
        type_num=job.cst.MessageType.CANCEL.value
    elif type =='market':
        type_num=job.cst.MessageType.MATCH.value
    else:
        raise ValueError('Type is none of: limit, cancel, delete or market')

    if side=='bid':
        side_num=job.cst.BidAskSide.BID.value
    elif side =='ask':
        side_num=job.cst.BidAskSide.ASK.value
    else:
        raise ValueError('Side is none of: bid or ask')
    
    dict_msg={
        'side':side_num,
        'type':type_num,
        'price':price,
        'quantity':quant,
        'orderid':id,
        'traderid':id,
        'time':times,
        'time_ns':timens}
    array_msg=jnp.array([type_num,side_num,quant,price,id,id,times,timens])
    return dict_msg,array_msg

@partial(jax.jit, static_argnums=(1,))
def get_random_order_to_cancel(book_side,
                               side='bid',
                               prev_time=job.cst.TEST_TIME,
                               times_range=[0,2],
                               timens_range=[0,job.cst.NS_PER_SEC]):
        """
        Obtains a random order ID from the given book side (asks or bids) and returns the order to cancel.
        """
        # Filter out empty slots
        valid_orders = book_side[book_side[:, job.cst.OrderSideFeat.OID.value] != job.cst.EMPTY_SLOT]
        
        
        if valid_orders.shape[0] == 0:
            raise ValueError("No valid orders to cancel.")
        

        delta_times = random.randint(times_range[0], times_range[1])
        delta_timens = random.randint(timens_range[0], timens_range[1])
        
        # Select a random order
        random_index = random.randint(0, valid_orders.shape[0] - 1)
        order_to_cancel = valid_orders[random_index]
        
        return create_message(type='cancel',
                              side=side,
                              price=order_to_cancel[job.cst.OrderSideFeat.P.value],
                              quant=order_to_cancel[job.cst.OrderSideFeat.Q.value],
                              times=prev_time[0]+delta_times,
                              timens=prev_time[1]+delta_timens,
                              id=order_to_cancel[job.cst.OrderSideFeat.OID.value])

def get_random_aggressive_order(book_side,
                                side='bid',
                                prev_time=job.cst.TEST_TIME,
                                times_range=[0,2],
                                timens_range=[0,job.cst.NS_PER_SEC]):
        """
        Obtains a price that will guarantee to match, randomly selects a quantity to match which is between 0 and 2x the quantity of the order to match.
        """
        
        if side=='bid':
            best_price=job.get_best_bid(job.Configuration(),book_side)
            best_vol=job.get_volume_at_price(book_side,best_price)
        else:
            best_price=job.get_best_ask(job.Configuration(),book_side)
            best_vol=job.get_volume_at_price(book_side,best_price)
        
        delta_times = random.randint(times_range[0], times_range[1])
        delta_timens = random.randint(timens_range[0], timens_range[1])

        opp_side= 'ask' if side=='bid' else 'bid'
        
        return create_message(type='limit',
                              side=opp_side,
                              price=best_price,
                              quant= random.randint(0,2*best_vol),
                              times=prev_time[0]+delta_times,
                              timens=prev_time[1]+delta_timens,
                              id=job.cst.DUMMYID)

def create_message_forvmap(type='limit',side='bid',price=2200000,quant=10,times=36000,timens=0,nvmap=10):
    if type=='limit':
        type_num=1
    elif type =='cancel' or type == 'delete':
        type_num=2
    elif type =='market':
        type_num=4
    else:
        raise ValueError('Type is none of: limit, cancel, delete or market')

    if side=='bid':
        side_num=1
    elif side =='ask':
        side_num=-1
    else:
        raise ValueError('Side is none of: bid or ask')
    
    dict_msg={
    'side':jnp.array([side_num]*nvmap),
    'type':jnp.array([type_num]*nvmap),
    'price':jnp.array([price]*nvmap),
    'quantity':jnp.array([quant]*nvmap),
    'orderid':jnp.array([8888]*nvmap),
    'traderid':jnp.array([8888]*nvmap),
    'time':jnp.array([times]*nvmap),
    'time_ns':jnp.array([timens]*nvmap)}
    array_msg=jnp.array([type_num,side_num,quant,price,8888,8888,times,timens]*nvmap)
    return dict_msg,array_msg


if __name__ == "__main__":
    # Example configuration object

    cfg = job.Configuration()

    key=jax.random.PRNGKey(42)
    key,subkey=jax.random.split(key)
    # Create initial order book
    asks, bids, trades = create_init_book(cfg)

    # Print the initial order book
    print("Initial Asks:\n", asks)
    print("Initial Bids:\n", bids)
    print("Initial Trades:\n", trades)

    print("Random order to cancel",get_random_aggressive_order(bids))

    # # Create a specific message
    # dict_msg, array_msg = create_message()
    # print("\nSpecific Message (Dict):\n", dict_msg)
    # print("Specific Message (Array):\n", array_msg)

    # # Create messages for vmap
    # dict_msg_vmap, array_msg_vmap = create_message_forvmap()
    # print("\nMessages for vmap (Dict):\n", dict_msg_vmap)
    # print("Messages for vmap (Array):\n", array_msg_vmap)