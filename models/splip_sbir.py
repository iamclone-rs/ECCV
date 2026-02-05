import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


def _is_ln(m):
    return isinstance(m, nn.LayerNorm)


def freeze_all_except_ln(m: nn.Module):
    for p in m.parameters():
        p.requires_grad_(False)
    for mod in m.modules():
        if _is_ln(mod):
            for p in mod.parameters(recurse=False):
                p.requires_grad_(True)


def infer_visual_dim(visual: nn.Module) -> int:
    if hasattr(visual, "class_embedding"):
        return int(getattr(visual, "class_embedding").shape[-1])
    if hasattr(visual, "positional_embedding"):
        return int(getattr(visual, "positional_embedding").shape[-1])
    if hasattr(visual, "conv1") and hasattr(visual.conv1, "out_channels"):
        return int(visual.conv1.out_channels)
    if hasattr(visual, "width"):
        return int(getattr(visual, "width"))
    # NOTE: For OpenCLIP ViT models, attributes like `output_dim`/`embed_dim` are often the
    # *projected* CLIP embedding size (e.g., 512) while our token features use transformer width
    # (e.g., 768). Prefer token-width signals above.
    if hasattr(visual, "dim"):
        return int(getattr(visual, "dim"))
    if hasattr(visual, "embed_dim"):
        return int(getattr(visual, "embed_dim"))
    if hasattr(visual, "output_dim"):
        return int(getattr(visual, "output_dim"))
    raise AttributeError("Cannot infer visual embedding dim (dv) from CLIP visual module")


class Bt(nn.Module):
    def __init__(self, dv, dt, m=4, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = max(dv, dt)
        self.m = m
        self.net = nn.Sequential(
            nn.Linear(dv, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, m * dt),
        )

    def forward(self, E0):
        x = E0.mean(dim=1)
        return self.net(x).view(x.size(0), self.m, -1)


class Bv(nn.Module):
    def __init__(self, dt, dv):
        super().__init__()
        self.proj = nn.Linear(dt, dv)

    def forward(self, W_prime):
        return self.proj(W_prime)


class Bvt(nn.Module):
    def __init__(self, dt, dv, n=2, bottleneck=256):
        super().__init__()
        self.n = n
        self.net = nn.Sequential(
            nn.Linear(dt, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, n * dv),
        )

    def forward(self, Wl_all):
        x = Wl_all.mean(dim=(0, 1))
        return self.net(x).view(self.n, -1)


class SpLIP_SBIR(nn.Module):
    def __init__(
        self,
        clip_name="ViT-B-32",
        pretrained="openai",
        m=4,
        n=2,
        bvt_bottleneck=256,
        device=None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.clip = self.clip.to(self.device)

        self.visual = self.clip.visual
        self.text_tr = self.clip.transformer
        self.token_embedding = self.clip.token_embedding
        self.pos_embedding = self.clip.positional_embedding
        self.ln_final = self.clip.ln_final
        self.text_proj = self.clip.text_projection

        dv = infer_visual_dim(self.visual)
        dt = self.token_embedding.embedding_dim

        self.m = m
        self.n = n

        self.bt = Bt(dv, dt, m=m)
        self.bv = Bv(dt, dv)
        self.bvt = Bvt(dt, dv, n=n, bottleneck=bvt_bottleneck)

        freeze_all_except_ln(self.clip)
        for p in self.bt.parameters():
            p.requires_grad_(True)
        for p in self.bv.parameters():
            p.requires_grad_(True)
        for p in self.bvt.parameters():
            p.requires_grad_(True)

        self._cache_seen = {}

    @torch.no_grad()
    def _get_E0(self, images):
        v = self.visual
        x = v.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        return x

    def _text_forward_deep_bt(self, tokenized, T):
        B, M = tokenized.shape
        x = self.token_embedding(tokenized)
        x = x + self.pos_embedding[:M, :]
        x = x.permute(1, 0, 2)

        Wl = []
        for blk in self.text_tr.resblocks:
            x = x.permute(1, 0, 2)
            x = torch.cat([T, x[:, self.m:, :]], dim=1)
            x = x.permute(1, 0, 2)
            x = blk(x)
            Wl.append(x.permute(1, 0, 2))

        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eot = tokenized.argmax(dim=-1)
        cls = x[torch.arange(B, device=x.device), eot]
        cls = cls @ self.text_proj
        return F.normalize(cls, dim=-1), Wl

    @torch.no_grad()
    def encode_text_plain(self, tokenized):
        x = self.token_embedding(tokenized)
        x = x + self.pos_embedding[: x.size(1), :]
        x = x.permute(1, 0, 2)
        x = self.text_tr(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eot = tokenized.argmax(dim=-1)
        x = x[torch.arange(x.size(0), device=x.device), eot] @ self.text_proj
        return F.normalize(x, dim=-1)

    def _visual_forward_deep(self, images, Vtg, Vms_per_layer):
        v = self.visual
        x = v.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)

        B = x.size(0)
        for i, blk in enumerate(v.transformer.resblocks):
            Vms = Vms_per_layer[i].unsqueeze(0).expand(B, -1, -1)
            Vi = torch.cat([Vtg, Vms], dim=1)
            x = torch.cat([x[:, :1, :], Vi, x[:, 1:, :]], dim=1)
            x = x.permute(1, 0, 2)
            x = blk(x)
            x = x.permute(1, 0, 2)

        x = v.ln_post(x[:, 0, :])
        if getattr(v, "proj", None) is not None:
            x = x @ v.proj
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def build_seen_prompt_tokens(self, class_names_seen, templates):
        key = (tuple(class_names_seen), templates["photo"], templates["sketch"], self.device)
        if key in self._cache_seen:
            return self._cache_seen[key]
        ph = [templates["photo"].format(c) for c in class_names_seen]
        sk = [templates["sketch"].format(c) for c in class_names_seen]
        tok_ph = self.tokenizer(ph).to(self.device)
        tok_sk = self.tokenizer(sk).to(self.device)
        self._cache_seen[key] = (tok_ph, tok_sk)
        return tok_ph, tok_sk

    @torch.no_grad()
    def _compute_Vtg_Vms_seen(self, proto_E0, class_names_seen, templates):
        tok_seen_ph, _ = self.build_seen_prompt_tokens(class_names_seen, templates)
        Cs = tok_seen_ph.size(0)

        proto_E0 = proto_E0.expand(Cs, -1, -1)
        T_proto = self.bt(proto_E0)

        _, Wl_seen_ph = self._text_forward_deep_bt(tok_seen_ph, T_proto)
        Wl_seen_ph = [w.detach() for w in Wl_seen_ph]

        W0 = self.token_embedding(tok_seen_ph).detach()
        W_prime = W0[:, 1:1 + self.m, :]
        Vtg_single = self.bv(W_prime)
        Vtg_anchor = Vtg_single.mean(dim=0)

        Vms_per_layer = []
        L = len(self.visual.transformer.resblocks)
        for l in range(L):
            Vms_per_layer.append(self.bvt(Wl_seen_ph[l]))

        return Vtg_anchor, Vms_per_layer, tok_seen_ph

    def forward_train(
        self,
        sketch,
        photo_pos,
        photo_neg,
        y_pos,
        y_neg,
        class_names_seen,
        templates,
        tau=0.07,
        alpha=1.0,
    ):
        B = sketch.size(0)
        device = sketch.device

        E0_pp = self._get_E0(photo_pos)
        proto_E0 = E0_pp.mean(dim=0, keepdim=True)

        Vtg_anchor, Vms_per_layer, tok_seen_ph = self._compute_Vtg_Vms_seen(
            proto_E0, class_names_seen, templates
        )
        Vtg = Vtg_anchor.unsqueeze(0).expand(B, -1, -1)

        z_sk = self._visual_forward_deep(sketch, Vtg, Vms_per_layer)
        z_pp = self._visual_forward_deep(photo_pos, Vtg, Vms_per_layer)
        z_pn = self._visual_forward_deep(photo_neg, Vtg, Vms_per_layer)

        class_embeds = self.encode_text_plain(tok_seen_ph)

        c_pos = class_embeds[y_pos]
        c_neg = class_embeds[y_neg]
        mu = (c_pos * c_neg).sum(dim=-1)

        d_pos = ((z_sk - z_pp) ** 2).sum(dim=-1)
        d_neg = ((z_sk - z_pn) ** 2).sum(dim=-1)
        l_triplet = F.relu(d_pos - d_neg + mu).mean()

        logits_sk = (z_sk @ class_embeds.t()) / tau
        logits_pp = (z_pp @ class_embeds.t()) / tau
        l_class = 0.5 * (F.cross_entropy(logits_sk, y_pos) + F.cross_entropy(logits_pp, y_pos))

        loss = l_triplet + alpha * l_class
        return loss, {
            "l_triplet": l_triplet.detach(),
            "l_class": l_class.detach(),
            "mu_mean": mu.mean().detach(),
        }

    @torch.no_grad()
    def encode_image_for_retrieval(self, images, class_names_seen, templates):
        B = images.size(0)
        E0 = self._get_E0(images)
        proto_E0 = E0.mean(dim=0, keepdim=True)

        Vtg_anchor, Vms_per_layer, _ = self._compute_Vtg_Vms_seen(
            proto_E0, class_names_seen, templates
        )
        Vtg = Vtg_anchor.unsqueeze(0).expand(B, -1, -1)
        return self._visual_forward_deep(images, Vtg, Vms_per_layer)
