"""Load pretrained encoder weights into a fresh DenseHMT for fine-tuning."""
import torch


def load_encoder(model, ckpt_path, freeze=False, verbose=True):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("encoder", sd)
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in ok]
    model.load_state_dict(ok, strict=False)
    if verbose:
        print(f"transferred {len(ok)}/{len(own)} tensors; not loaded: "
              f"{len(missing)} (expected: head.*)")
    if freeze:
        for n, p in model.named_parameters():
            if not n.startswith("head."):
                p.requires_grad = False
        print("encoder frozen; training head only")
    return model
