import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class CNN_RNN(nn.Module):
    def __init__(self, cnn_params={}, rnn_params={}):
        super().__init__()
        
        self.max_sequence = cnn_params.get("max_sequence", 350)
        self.num_hand_classes = cnn_params.get("num_hand_classes", 11)
        self.num_note_classes = cnn_params.get("num_note_classes", 128)
        self.note_weight = cnn_params.get("note_weight", 1.0)
        self.time_stamp_weight = cnn_params.get("time_stamp_weight", 0.01)
        self.hand_weight = cnn_params.get("hand_weight", 1.0)
        self.coord_weight = cnn_params.get("coord_weight", 0.1)
        self.rnn_weight = cnn_params.get("rnn_weight", 1.0)
        
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        old_conv = mobilenet.features[0][0]
        new_conv = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
        mobilenet.features[0][0] = new_conv
        
        self.cnn = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        cnn_out_features = 576
        
        self.hand_classifier = nn.Linear(cnn_out_features, self.max_sequence * self.num_hand_classes)
        self.note_classifier = nn.Linear(cnn_out_features, self.max_sequence * self.num_note_classes)
        self.time_regressor = nn.Linear(cnn_out_features, self.max_sequence)
        self.coord_regressor = nn.Linear(cnn_out_features, self.max_sequence * 4)
        
        rnn_input_size = 3
        self.rnn_hidden_size = rnn_params.get("hidden_size", 32)
        rnn_output_size = rnn_params.get("output_size", self.num_note_classes)
        rnn_layers = rnn_params.get("num_layers", 2)
        
        self.input_linear = nn.Linear(rnn_input_size, self.rnn_hidden_size)
        self.rnn = nn.LSTM(
            input_size=self.rnn_hidden_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.classifier = nn.Linear(self.rnn_hidden_size * 2, rnn_output_size)

    def forward(self, x, y=None, coords_target=None):
        device = next(self.parameters()).device
        x = x.to(device)
        if y is not None:
            y = y.to(device)
        if coords_target is not None:
            coords_target = coords_target.to(device)
        
        features = self.cnn(x)
        pooled = self.pool(features).view(features.size(0), -1)
        
        hand_logits = self.hand_classifier(pooled).view(-1, self.max_sequence, self.num_hand_classes)
        note_logits = self.note_classifier(pooled).view(-1, self.max_sequence, self.num_note_classes)
        time_preds = self.time_regressor(pooled).view(-1, self.max_sequence)
        coord_preds = self.coord_regressor(pooled).view(-1, self.max_sequence, 4)
        
        loss = None
        if y is not None:
            loss_hand = F.cross_entropy(
                hand_logits.reshape(-1, self.num_hand_classes),
                y[:, 0, :].reshape(-1).long(),
                ignore_index=999
            ) * self.hand_weight
            
            loss_note = F.cross_entropy(
                note_logits.reshape(-1, self.num_note_classes),
                y[:, 1, :].reshape(-1).long(),
                ignore_index=999
            ) * self.note_weight
            
            valid_mask = (y[:, 2, :] != 0).float()
            if valid_mask.sum() > 0:
                loss_time = (F.mse_loss(time_preds, y[:, 2, :], reduction='none') * valid_mask).sum() / valid_mask.sum() * self.time_stamp_weight
            else:
                loss_time = torch.tensor(0.0, device=device)
            
            loss = loss_hand + loss_note + loss_time
            
            if coords_target is not None:
                valid_coord_mask = (coords_target.sum(dim=-1) != 0).float()
                if valid_coord_mask.sum() > 0:
                    loss_coord = (F.mse_loss(coord_preds, coords_target, reduction='none').mean(dim=-1) * valid_coord_mask).sum() / valid_coord_mask.sum() * self.coord_weight
                    loss = loss + loss_coord
        
        logits = {
            "cnn_hand": hand_logits,
            "cnn_note": note_logits,
            "cnn_time": time_preds,
            "cnn_coords": coord_preds
        }
        
        return loss, logits

    def forward_rnn(self, seq_inputs):
        seq_inputs = seq_inputs.float().to(next(self.parameters()).device)
        rnn_in = self.input_linear(seq_inputs)
        lstm_out, _ = self.rnn(rnn_in)
        lstm_out = F.dropout(lstm_out, p=0.4, training=self.training)
        rnn_logits = self.classifier(lstm_out)
        return rnn_logits