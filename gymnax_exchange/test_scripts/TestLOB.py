from functools import partial, partialmethod
from typing import OrderedDict
from jax import numpy as jnp
import jax
import numpy as np
import random
import time
import timeit

import sys
# ******** INSERT PATH HERE ********
sys.path.append('/home/duser/AlphaTrade/')
import gymnax_exchange
import gymnax_exchange.jaxob.JaxOrderBookArrays as job
import gymnax_exchange.utils.utils as utils



class TestLimitOrderBookSimulator:
    def test_add_order_to_full_book(self):
        # Assuming the order book has a maximum size of 5 for this example
        max_size = 5
        for i in range(max_size):
            self.simulator.add_order({"id": i, "type": "buy", "price": 100 + i, "quantity": 10})

        # Try to add one more order beyond the maximum size
        extra_order = {"id": max_size, "type": "buy", "price": 110, "quantity": 10}
        try:
            self.simulator.add_order(extra_order)
            assert False, "Expected an exception when adding an order to a full book"
        except Exception as e:
            assert str(e) == "Order book is full", f"Unexpected exception message: {str(e)}"
    def setup_method(self):
        self.simulator = None

    def test_add_order(self):
        pass

    def test_cancel_order(self):
        pass

    def test_match_orders(self):
        pass

    def test_get_order_book(self):
        pass


class SpeedExperimentsCore:
    def __init__(self):
        self.simulator = None
        self.cfg = job.Configuration()
        self.key=jax.random.PRNGKey(42)

    def bootstrap_confidence_interval(self, data, num_samples=1000, confidence_level=0.99):
        sample_means = []
        n = len(data)
        for _ in range(num_samples):
            sample = [random.choice(data) for _ in range(n)]
            sample_means.append(np.mean(sample))
        lower_bound = np.percentile(sample_means, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(sample_means, (1 + confidence_level) / 2 * 100)
        return lower_bound, upper_bound

    def test_speed_add_order(self,n_orders,booksize,rand_orders=True,n_samples=100):
        times = []
        
        for _ in range(n_samples):
            asks,bids,trades=utils.create_init_book(self.cfg,order_capacity=booksize,trade_capacity=booksize)
            orders=[]
            for _ in range(n_orders):
                mdict,marray=utils.create_rand_message(type='limit',side='bid')
                orders.append(mdict)
            if rand_orders:
                start_time = time.time()
                for order in orders:
                    bids=job.add_order(bids,order)
                end_time = time.time()
                times.append(end_time - start_time)
            else:
                start_time = time.time()
                for order in orders:
                    out=job.add_order(bids,orders[0])
                end_time = time.time()
                times.append(end_time - start_time)

        # Remove the first time due to compilation overhead.
        times=times[1:]
        lower_bound, upper_bound = self.bootstrap_confidence_interval(times)
        mean_time = np.mean(times)
        print(f"Mean time for adding {len(orders)} orders: {mean_time} seconds")
        print(f"99% confidence interval for adding {len(orders)} orders: [{lower_bound}, {upper_bound}] seconds")
        return mean_time, lower_bound, upper_bound

    def test_speed_match_orders(self,n_orders,booksize,find_order=True,match_order=True,n_samples=100):
        times = []
        cfg=job.Configuration()
        
        for _ in range(n_samples):
            asks,bids,trades=utils.create_init_book(self.cfg,order_capacity=booksize,trade_capacity=booksize)
            orders=[]
            for _ in range(n_orders):
                mdict,marray=utils.get_random_aggressive_order(bids,side='bid')
                orders.append(mdict)
            
            if find_order and match_order:
                start_time = time.time()
                for order in orders:
                    out,qtm,price,trade=job._match_against_bid_orders(cfg,bids,order["quantity"],order["price"],trades,order["orderid"],order["time"],order["time_ns"],order["orderid"],job.cst.BidAskSide.BID.value)
                end_time = time.time()
                times.append(end_time - start_time)
            if find_order:
                start_time = time.time()
                for order in orders:
                    out=job._get_top_bid_order_idx(cfg,bids)
                end_time = time.time()
                times.append(end_time - start_time)
            if match_order:
                matchtuples=[]
                for order in orders:
                    idx=job._get_top_bid_order_idx(cfg,bids)
                    matchtuples.append((idx,order["quantity"],order["price"],trades,order["orderid"],order["time"],order["time_ns"],order["orderid"],job.cst.BidAskSide.BID.value))
                start_time = time.time()
                for matchtuple in matchtuples:
                    out=job.match_order(matchtuple)
                end_time = time.time()
                times.append(end_time - start_time)

        # Remove the first time due to compilation overhead.
        times=times[1:]
        lower_bound, upper_bound = self.bootstrap_confidence_interval(times)
        mean_time = np.mean(times)
        print(f"Mean time for adding {len(orders)} orders: {mean_time} seconds")
        print(f"99% confidence interval for adding {len(orders)} orders: [{lower_bound}, {upper_bound}] seconds")
        return mean_time, lower_bound, upper_bound

    def test_speed_cancel_order(self,n_orders,booksize,n_samples=100):
        times = []
        for _ in range(n_samples):
            asks, bids, trades = utils.create_init_book(self.cfg,order_capacity=booksize, trade_capacity=booksize)
            order_dict,_ = utils.get_random_order_to_cancel(bids,side='bid')
            start_time = time.time()
            for _ in range(n_orders):
                out=job.cancel_order(self.cfg, self.key,bids,order_dict)
            end_time = time.time()
            times.append(end_time - start_time)
        # Remove the first time due to compilation overhead.
        times = times[1:]
        lower_bound, upper_bound = self.bootstrap_confidence_interval(times)
        mean_time = np.mean(times)
        print(f"Mean time for canceling {n_orders} orders: {mean_time} seconds")
        print(f"99% confidence interval for canceling {n_orders} orders: [{lower_bound}, {upper_bound}] seconds")
        return mean_time, lower_bound, upper_bound

    def test_speed_get_order_book(self):
        for i in range(1000):
            self.simulator.add_order({"id": i, "type": "buy", "price": 100 + i, "quantity": 10})
        start_time = time.time()
        self.simulator.get_order_book()
        end_time = time.time()
        print(f"Time taken to get order book: {end_time - start_time} seconds")

if __name__ == "__main__":
    tester = TestLimitOrderBookSimulator()
    tester.setup_method()
    tester.test_add_order()
    tester.test_cancel_order()
    tester.test_match_orders()
    tester.test_get_order_book()

    speed_tester = SpeedExperimentsCore()
    # speed_tester.test_speed_add_order(1000,100,rand_orders=True)
    speed_tester.test_speed_match_orders(100,100)

