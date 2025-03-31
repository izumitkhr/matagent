from .categorical import CategoricalTransition, GeneralCategoricalTransition
from .continuous import ContinuousTransition
from .wrapped_normal import WNtransition


def get_transition(cfg, scheduler, num_classes=100):
    type = cfg.type
    if type == "wn":
        return WNtransition(scheduler)
    elif type == "continuous":
        return ContinuousTransition(scheduler, num_classes)
    elif type == "categorical":
        return CategoricalTransition(scheduler, num_classes)
    elif type == "general_categorical":
        init_prob = cfg.init_prob
        return GeneralCategoricalTransition(scheduler, num_classes, init_prob)
    else:
        raise ValueError(f"Unknown type: {type}")
