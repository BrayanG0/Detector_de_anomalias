import torch
import torch.nn as nn
import torchvision


def _proj(in_f, out_f=128):
    return nn.Sequential(nn.Linear(in_f, out_f), nn.ReLU(), nn.Dropout(0.3))


class RedColaborativaMURA(nn.Module):
    """4 backbones preentrenados -> proyección a 128 -> atención -> clasificador.
    Devuelve LOGITS (sin sigmoid). Usar BCEWithLogitsLoss."""

    def __init__(self, pretrained=True, proj_dim=128, full_finetune=False, large=False):
        super().__init__()
        w = "DEFAULT" if pretrained else None

        self.densenet = (torchvision.models.densenet169 if large else torchvision.models.densenet121)(weights=w)
        n_dense = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()

        self.resnet = (torchvision.models.resnet101 if large else torchvision.models.resnet50)(weights=w)
        n_res = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.efficientnet = (torchvision.models.efficientnet_b3 if large else torchvision.models.efficientnet_b0)(weights=w)
        n_eff = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        self.mobilenet = torchvision.models.mobilenet_v2(weights=w)
        n_mob = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Identity()

        # Congelar todo y descongelar solo los últimos bloques
        for b in self.backbones():
            for p in b.parameters():
                p.requires_grad = False
        unfreeze = [
            self.densenet.features.denseblock4,
            self.densenet.features.norm5,
            self.resnet.layer4,
            self.efficientnet.features[-2:],
            self.mobilenet.features[-3:],
        ]
        for m in unfreeze:
            for p in m.parameters():
                p.requires_grad = True

        if full_finetune:  # entrenar TODAS las capas de los 4 backbones
            for b in self.backbones():
                for p in b.parameters():
                    p.requires_grad = True

        self.proj_dense = _proj(n_dense, proj_dim)
        self.proj_res = _proj(n_res, proj_dim)
        self.proj_eff = _proj(n_eff, proj_dim)
        self.proj_mob = _proj(n_mob, proj_dim)

        self.atencion = nn.Sequential(nn.Linear(proj_dim * 4, 4), nn.Softmax(dim=1))
        self.clasificador = nn.Sequential(
            nn.Linear(proj_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # logit
        )

    def backbones(self):
        return [self.densenet, self.resnet, self.efficientnet, self.mobilenet]

    def train(self, mode=True):
        """Mantiene en eval() los BatchNorm de las capas congeladas."""
        super().train(mode)
        if mode:
            for b in self.backbones():
                for m in b.modules():
                    if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
                        m.eval()
        return self

    def param_groups(self, lr_backbone=1e-5, lr_head=1e-4):
        bb = [p for b in self.backbones() for p in b.parameters() if p.requires_grad]
        bb_ids = {id(p) for p in bb}
        head = [p for p in self.parameters() if p.requires_grad and id(p) not in bb_ids]
        return [
            {"params": bb, "lr": lr_backbone},
            {"params": head, "lr": lr_head},
        ]

    def forward(self, x):
        return self.forward_con_pesos(x)[0]

    def forward_con_pesos(self, x):
        """Devuelve (logit, pesos de atención de los 4 modelos)."""
        f_d = self.proj_dense(self.densenet(x))
        f_r = self.proj_res(self.resnet(x))
        f_e = self.proj_eff(self.efficientnet(x))
        f_m = self.proj_mob(self.mobilenet(x))

        pesos = self.atencion(torch.cat([f_d, f_r, f_e, f_m], dim=1))
        fusion = torch.cat([
            f_d * pesos[:, 0:1],
            f_r * pesos[:, 1:2],
            f_e * pesos[:, 2:3],
            f_m * pesos[:, 3:4],
        ], dim=1)
        return self.clasificador(fusion), pesos