from collections import OrderedDict


def keep_encoder_only(ckpt):
    """
    Finds the end of the encoder (encoder.norm.weight) and removes
    all keys that come after it, as they are useless for finetuning.
    """

    end_index = [i for i,j in enumerate(list(ckpt.keys())) if j=='encoder.norm.weight'][0]

    for k in list(ckpt.keys())[end_index:]:
        ckpt.pop(k, None)
    
    return ckpt


def remap_checkpoint_keys(ckpt):
    """
    This function helps harmonize key names in saved weights dictionary in order to
    transfer the weights to another model.
    """

    ckpt = keep_encoder_only(ckpt)

    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("encoder"):
            k = ".".join(k.split(".")[1:])  # remove encoder in the name
        if k.endswith("kernel"):
            k = ".".join(k.split(".")[:-1])  # remove kernel in the name
            new_k = k + ".weight"
            continue
        elif "ln" in k or "linear" in k:
            k = k.split(".")
            k.pop(-2)  # remove ln and linear in the name
            new_k = ".".join(k)
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt
