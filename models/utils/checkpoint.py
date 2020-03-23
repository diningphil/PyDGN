from copy import deepcopy


def state_dict(model, optimizer):
    # print('Checkpointing...')
    return deepcopy({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()})