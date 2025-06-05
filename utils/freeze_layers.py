from collections.abc import Iterable


def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(list(model.children())):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)


def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)