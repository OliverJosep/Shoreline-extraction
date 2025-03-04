import torch
import torch.nn as nn

class BiLSTM_architecture(nn.Module):
    def __init__(self,  in_channels = 3, out_channels = 1, hidden_units = 45):
        super(BiLSTM_architecture, self).__init__()
        self.out_channels = out_channels
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_units, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_units*2, out_channels)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)

        if self.out_channels > 1:
            return torch.softmax(out, dim=1) # multi-class segmentation
        else:
            # return torch.sigmoid(logits) # binary segmentation
            return out   # binary segmentation