from copy import deepcopy

from pydgn.static import *


def state_dict(model, optimizer):
    # print('Checkpointing...')
    return deepcopy({MODEL_STATE_DICT: model.state_dict(), OPTIMIZER_STATE_DICT: optimizer.state_dict()})
