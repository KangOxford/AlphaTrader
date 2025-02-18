import gymnax_exchange.jaxob.jaxob_constants as cst
import jax
from typing import Tuple

from dataclasses import dataclass

@dataclass(frozen=True)
class Configuration:
    maxint : int = cst.MaxInt._64_Bit_Signed.value
    init_id :int = cst.INITID
    cancel_mode: int= cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value
    seed: int =cst.SEED
    nTrades : int=cst.NTRADE_CAP
    nOrders : int =cst.NORDER_CAP
    simulator_mode=cst.SimulatorMode.GENERAL_EXCHANGE.value
    empty_slot_val=cst.EMPTY_SLOT

    
    