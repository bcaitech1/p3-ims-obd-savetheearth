import torch

_scheduler_entrypoints = {
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR,
}

def scheduler_entrypoint(scheduler_name):
    return _scheduler_entrypoints[scheduler_name]


def is_scheduler(scheduler_name):
    return scheduler_name in _scheduler_entrypoints


def create_scheduler(scheduler_name, **kwargs):
    if is_scheduler(scheduler_name):
        create_fn = scheduler_entrypoint(scheduler_name)
        scheduler = create_fn(**kwargs)
        
    else:
        raise RuntimeError('Unknown scheduler (%s)' % scheduler_name)
        
    return scheduler