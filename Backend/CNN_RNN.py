import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_RNN(nn.Module):
    def __init__(self, cnn_params={}, rnn_params={}):
        super().__init__()

        # --- CNN part ---
        input_channels = cnn_params.get("input_channels", 3)
        self.max_sequence = cnn_params.get("max_sequence", 350)
        self.num_hand_classes = cnn_params.get("num_hand_classes", 10)
        self.num_note_classes = cnn_params.get("num_note_classes", 128)
        self.note_weight = cnn_params.get("note_weight", 1.0)
        self.time_stamp_weight = cnn_params.get("time_stamp_weight", 0.01)
        self.hand_weight = cnn_params.get("hand_weight", 1.0)
        self.rnn_weight = cnn_params.get("rnn_weight", 1.0)

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((7, 7))
        )

        # CNN classifiers
        self.hand_classifier = nn.Linear(256 * 7 * 7, self.max_sequence * self.num_hand_classes)
        self.note_classifier = nn.Linear(256 * 7 * 7, self.max_sequence * self.num_note_classes)
        self.time_regressor = nn.Linear(256 * 7 * 7, self.max_sequence)

        # --- RNN part ---
        rnn_input_size = 256            # from CNN features
        rnn_hidden_size = rnn_params.get("hidden_size", 512)
        rnn_output_size = rnn_params.get("output_size", self.num_note_classes)
        rnn_layers = rnn_params.get("num_layers", 3)

        self.input_linear = nn.Linear(rnn_input_size, rnn_hidden_size)
        self.rnn = nn.LSTM(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(rnn_hidden_size * 2, rnn_output_size)

    # ------------------------------
    # Helper: map labels to RNN sequence length
    # ------------------------------
    def map_labels_to_rnn(self, labels, rnn_seq_len=49, pad_value=999):
        """
        labels: tensor [batch, MAX_NOTES]
        rnn_seq_len: target sequence length for RNN
        """
        batch_size, max_notes = labels.shape
        if max_notes < rnn_seq_len:
            # pad if sequence is too short
            padded = torch.full((batch_size, rnn_seq_len), pad_value, dtype=labels.dtype, device=labels.device)
            padded[:, :max_notes] = labels
            return padded
        else:
            # uniform downsampling
            idx = torch.linspace(0, max_notes-1, steps=rnn_seq_len).long()
            return labels[:, idx]

    # ------------------------------
    # Forward
    # ------------------------------
    def forward(self, x, y=None):
        x = x.to(self.cnn[0].weight.device)
        if y is not None:
            y = y.to(self.cnn[0].weight.device)

        # --- CNN feature extraction ---
        features = self.cnn(x)                # [batch, 256, 7, 7]
        flattened = features.view(features.size(0), -1)

        # CNN classifiers
        hand_logits = self.hand_classifier(flattened).view(-1, self.max_sequence, self.num_hand_classes)
        note_logits = self.note_classifier(flattened).view(-1, self.max_sequence, self.num_note_classes)
        time_preds = self.time_regressor(flattened).view(-1, self.max_sequence)

        # --- Prepare RNN input ---
        batch_size = features.size(0)
        cnn_seq = features.view(batch_size, 256, -1).permute(0, 2, 1)  # [batch, 49, 256]
        rnn_in = self.input_linear(cnn_seq)
        lstm_out, _ = self.rnn(rnn_in)
        rnn_logits = self.classifier(lstm_out)  # [batch, 49, output_size]

        # --- Align labels for RNN ---
        labels_rnn = None
        loss = None
        if y is not None:
            hand_rnn = self.map_labels_to_rnn(y[:, 0, :], rnn_seq_len=cnn_seq.size(1))
            note_rnn = self.map_labels_to_rnn(y[:, 1, :], rnn_seq_len=cnn_seq.size(1))
            time_rnn = self.map_labels_to_rnn(y[:, 2, :], rnn_seq_len=cnn_seq.size(1))
            labels_rnn = torch.stack([hand_rnn, note_rnn, time_rnn], dim=1)

            # --- CNN losses ---
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
            loss_time = (F.mse_loss(time_preds, y[:, 2, :], reduction='none') * valid_mask).sum() / valid_mask.sum() * self.time_stamp_weight

            # --- RNN loss using aligned labels ---
            rnn_loss = F.cross_entropy(
                rnn_logits.reshape(-1, rnn_logits.size(-1)),
                labels_rnn[:, 1, :].reshape(-1).long(),
                ignore_index=999
            ) * self.rnn_weight

            loss = loss_hand + loss_note + loss_time + rnn_loss

        logits = {
            "cnn_hand": hand_logits,
            "cnn_note": note_logits,
            "cnn_time": time_preds,
            "rnn_note": rnn_logits
        }

        return loss, logits
