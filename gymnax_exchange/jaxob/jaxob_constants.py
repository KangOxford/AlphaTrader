
from enum import Enum

class MaxInt(Enum):
    _64_Bit_Signed=2_147_483_647
    _32_Bit_Signed=32_767

# Actual constants, will never change.
INITID=-900000
DUMMYID=-888888
EMPTY_SLOT=-1

ORDERBOOK_FEAT=6
TRADE_FEAT=8
NS_PER_SEC=1e9

# Default values for the config.

NTRADE_CAP=100
NORDER_CAP=100
STARTOFDAY=[34200 , 0]
ENDOFDAY=[57600 , 0]
TEST_TIME=[44444,44444]


# LOBSTER Message types, 
class MessageType(Enum):
    LIMIT=1  
    CANCEL=2 
    DELETE=3 
    MATCH=4
    HIDDEN=5
    AUCTION=6
    HALT=7

class OrderSideFeat(Enum):
    P=0 #Price
    Q=1 #Quantity
    OID=2 #Order ID
    TID=3 # Trade ID
    SEC=4 #Seconds
    NSEC=5 #Nanoseconds 

class BidAskSide(Enum):
    BID=1
    ASK=-1


#Define as static...? Might solve 
class CancelMode(Enum):
    STRICT_BY_ID=0 #Cancel only if ID matches. 
    INCLUDE_INITS=1 #Cancel only init orders if ID does not match
    CANCEL_UNIFORM=2  #Pick a random order at the right price level to cancel 
    CANCEL_UNIFORM_AND_LARGE=3 # Unused for now

SEED= 42 # the meaning of life. 

#TODO: flag on behaviour of type market either limit or far touch. 

class SimulatorMode(Enum):
    GENERAL_EXCHANGE=0
    LOBSTER_INTERPRETER=1
