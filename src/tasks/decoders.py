"""Decoder heads.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train

log = src.utils.train.get_logger(__name__)


class Decoder(nn.Module):
    """This class doesn't do much but just signals the interface that Decoders are expected to adhere to
    TODO: is there a way to enforce the signature of the forward method?
    """

    def forward(self, x, **kwargs):
        """
        x: (batch, length, dim) input tensor
        state: additional state from the model backbone
        *args, **kwargs: additional info from the dataset

        Returns:
        y: output tensor
        *args: other arguments to pass into the loss function
        """
        return x

    def step(self, x):
        """
        x: (batch, dim)
        """
        return self.forward(x.unsqueeze(1)).squeeze(1)


class SequenceDecoder(Decoder):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last",
            conjoin_train=False, conjoin_test=False, plain_bidir_train=False,
            plain_bidir_eval=False
    ):
        super().__init__()

        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)
        self.attn_query = None
        if mode in ["attn_pool", "query_pool", "cls_pool", "masked_attn_pool"]:
            self.attn_query = nn.Parameter(torch.zeros(d_model))

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.plain_bidir_train = plain_bidir_train
        self.plain_bidir_eval = plain_bidir_eval

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model) or potentially (n_batch, l_seq, d_model, 2) if using rc_conjoin
        Returns: (n_batch, l_output, d_output)
        """
        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(1)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            def restrict(x_seq):
                """Use last l_output elements of sequence."""
                return x_seq[..., -l_output:, :]

        elif self.mode == "first":
            def restrict(x_seq):
                """Use first l_output elements of sequence."""
                return x_seq[..., :l_output, :]

        elif self.mode == "pool":
            def restrict(x_seq):
                """Pool sequence over a certain range"""
                L = x_seq.size(1)
                s = x_seq.sum(dim=1, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x_seq[..., -(l_output - 1):, ...].flip(1), dim=1)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(1)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x_seq.dtype, device=x_seq.device
                )
                s = s / denom
                return s

        elif self.mode == "masked_pool":
            assert lengths is not None, "lengths must be provided for masked_pool mode"
            assert l_output == 1, "masked_pool currently supports only l_output=1"

            def restrict(x_seq):
                raise RuntimeError("masked_pool handles the batch directly")

            pooled = []
            for x_seq, length in zip(torch.unbind(x, dim=0), lengths.view(-1)):
                # NT uses left padding, so real sequence tokens are the final `length` positions.
                length = int(length.item())
                pooled.append(x_seq[-length:, ...].mean(dim=0, keepdim=True))
            x = torch.stack(pooled, dim=0)
            squeeze = self.squeeze

        elif self.mode in ["attn_pool", "query_pool", "cls_pool", "masked_attn_pool"]:
            assert l_output == 1, f"{self.mode} currently supports only l_output=1"

            def restrict(x_seq):
                raise RuntimeError(f"{self.mode} handles the batch directly")

            scores = torch.einsum("bld,d->bl", x, self.attn_query)
            scores = scores / (x.size(-1) ** 0.5)
            if lengths is not None:
                positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
                # NT batches use left padding: the final `length` positions are real sequence tokens.
                valid_from = x.size(1) - lengths.view(-1, 1).to(device=x.device)
                mask = positions >= valid_from
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            elif self.mode == "masked_attn_pool":
                raise AssertionError("lengths must be provided for masked_attn_pool mode")
            weights = torch.softmax(scores, dim=1).unsqueeze(1)
            x = torch.bmm(weights, x)
            squeeze = self.squeeze

        elif self.mode == "sum":
            # TODO use same restrict function as pool case
            def restrict(x_seq):
                """Cumulative sum last l_output elements of sequence."""
                return torch.cumsum(x_seq, dim=-2)[..., -l_output:, :]
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"

            def restrict(x_seq):
                """Ragged aggregation."""
                # remove any additional padding (beyond max length of any sequence in the batch)
                return x_seq[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'masked_pool' | "
                "'attn_pool' | 'query_pool' | 'cls_pool' | 'masked_attn_pool' | 'sum' | 'ragged']"
            )

        # Restrict to actual length of sequence
        if self.mode in ["masked_pool", "attn_pool", "query_pool", "cls_pool", "masked_attn_pool"]:
            pass
        elif self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(1) == 1
            x = x.squeeze(1)

        if self.conjoin_train or (self.conjoin_test and not self.training):
            if x.dim() >= 3 and x.size(-1) == 2:
                x_fwd = x[..., 0]
                x_rc = x[..., 1]
            else:
                x_fwd, x_rc = x.chunk(2, dim=-1)
            x = self.output_transform(x_fwd)
            x_rc = self.output_transform(x_rc)
            x = (x + x_rc) / 2
        else:
            x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        x_fwd = self.output_transform(x.mean(dim=1))
        x_rc = self.output_transform(x.flip(dims=[1, 2]).mean(dim=1)).flip(dims=[1])
        x_out = (x_fwd + x_rc) / 2
        return x_out


class PairSequenceDecoder(Decoder):
    """Pool ref/alt sequence representations and classify their pair."""

    def __init__(
        self,
        d_model,
        d_output=None,
        l_output=None,
        mode="pool",
        hidden_mult=4,
        conjoin_train=False,
        conjoin_test=False,
        plain_bidir_train=False,
        plain_bidir_eval=False,
    ):
        super().__init__()
        del l_output
        d_pair = d_model * 2
        if d_output is None:
            self.output_transform = nn.Identity()
        else:
            self.output_transform = nn.Sequential(
                nn.Linear(d_pair, d_pair * hidden_mult),
                nn.SiLU(),
                nn.Linear(d_pair * hidden_mult, d_pair),
                nn.SiLU(),
                nn.Linear(d_pair, d_output),
            )
        self.mode = mode
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.plain_bidir_train = plain_bidir_train
        self.plain_bidir_eval = plain_bidir_eval

    def _pool(self, x):
        if self.mode == "pool":
            return x.mean(dim=1)
        if self.mode == "last":
            return x[:, -1, ...]
        if self.mode == "first":
            return x[:, 0, ...]
        raise NotImplementedError("PairSequenceDecoder supports mode in ['pool', 'last', 'first'].")

    def forward(self, x, state=None, **kwargs):
        del state, kwargs
        x = self._pool(x)
        if self.conjoin_train or (self.conjoin_test and not self.training):
            if x.dim() != 3:
                raise ValueError("Conjoined pair decoding expects pooled tensor with shape (batch, channels, 2).")
            x_fwd = self.output_transform(x[..., 0])
            x_rev = self.output_transform(x[..., 1])
            return 0.5 * (x_fwd + x_rev), {}
        return self.output_transform(x), {}


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Decoder,
    "id": nn.Identity,
    "linear": nn.Linear,
    "sequence": SequenceDecoder,
    "pair_sequence": PairSequenceDecoder,
}

model_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output"],
    "pair_sequence": ["d_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_state", "state_to_tensor"],
    "forecast": ["d_output"],
    "token": ["d_output"],
}

dataset_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output", "l_output"],
    "pair_sequence": ["d_output", "l_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_output"],
    "forecast": ["d_output", "l_output"],
    "token": ["d_output"],
}


def _instantiate(decoder, model=None, dataset=None):
    """Instantiate a single decoder"""
    if decoder is None:
        return None

    if isinstance(decoder, str):
        name = decoder
    else:
        name = decoder["_name_"]

    # Extract arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
    # Instantiate decoder
    obj = utils.instantiate(registry, decoder, *model_args, *dataset_args)
    return obj


def instantiate(decoder, model=None, dataset=None):
    """Instantiate a full decoder config, e.g. handle list of configs
    Note that arguments are added in reverse order compared to encoder (model first, then dataset)
    """
    decoder = utils.to_list(decoder)
    return U.PassthroughSequential(
        *[_instantiate(d, model=model, dataset=dataset) for d in decoder]
    )
