# Main Function for Entire Pipeline #

import enum

SYNTHETIC_DATA_GENERATION = 1
SYNTHETIC_TRAINING = 1
REAL_DATA_FINETUNING = 1
TESTING = 0

class PipelineStages(enum.Enum):
    SYNTHETIC_DATA_GENERATION = 1
    SYNTHETIC_TRAINING = 2
    REAL_DATA_FINETUNING = 3
    TESTING = 4
class RadioMode(enum.Enum):
    TX = 1
    RX = 2
class Modulation(enum.Enum):
    BPSK = 1
    QPSK = 2

class radio_manager:
    tx_radio = ''
    rx_radio = ''
    def __init__(self, tx_radio, rx_radio):
        self.tx_radio = tx_radio
        self.rx_radio = rx_radio

    def connect_radios(self, TX_ID, RX_ID):
        self.rx_radio = RX_ID
        self.tx_radio = TX_ID

    def establish_connection(self):
        # Code to establish the connection between the radios
        pass
    def send_heartbeat(self):
        pass

    def execute_test(self, payloads, model):
        # Code to execute the test with the given payloads
        pass

class radio: 
    mode = RadioMode.TX
    frequency = 915e6
    preamble = ''
    pilots = []
    def __init__(self, mode, frequency = 915e6, preamble = '', pilots = []):
        self.mode = mode
        self.frequency = frequency
        self.preamble = preamble
        self.pilots = pilots

class tx_radio(radio):
    payloads = []
    def __init__(self, mode, frequency = 915e6, preamble = '', pilots = [], payloads = []):
        super().__init__(mode, frequency, preamble, pilots)
        self.payloads = payloads
    def set_payloads(self, payloads):
        self.payloads = payloads
    def transmit_payloads(self):
        # transmit preamble

        # Loop
            # transmit pilots
            # transmit payload i
        pass 

class rx_radio(radio):
    received_payloads = []
    def __init__(self, mode, frequency = 915e6, preamble = '', pilots = []):
        super().__init__(mode, frequency, preamble, pilots)
    def listen_and_receive(self, modulation):
        # Code to listen and receive data
        # return received data to pass into model
        pass

class model:
    def __init__(self):
        pass



stage = PipelineStages.SYNTHETIC_DATA_GENERATION

match stage:
    case PipelineStages.SYNTHETIC_DATA_GENERATION:
        pass
    case PipelineStages.SYNTHETIC_TRAINING:
        pass
    case PipelineStages.REAL_DATA_FINETUNING:
        pass
    case PipelineStages.TESTING:
        pass