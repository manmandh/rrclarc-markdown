import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention Gate (AG)
class AttentionGate(nn.Module):
  def __init__(self, in_channels, gating_channels, inter_channels):
    super(AttentionGate, self).__init__()
    # Điều chỉnh số lượng kênh đầu vào và kênh gating sao cho phù hợp
    self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
    self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
    self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)

  def forward(self, x, g):
    """
    :param x: feature maps (input image features)
    :param g: gating signal (from coarser scale)
    :return: output with attention applied
    """
    # Apply Conv1x1 on x and g
    phi_x = self.W_x(x)  # (b, c', h, w)
    phi_g = self.W_g(g)  # (b, c', h, w)

    # Tính attention map: dùng sigmoid để tính hệ số chú ý alpha
    attention = F.relu(phi_x + phi_g)  # (b, c', h, w)
    attention = self.psi(attention)    # (b, 1, h, w)
    attention = torch.sigmoid(attention)  # (b, 1, h, w)

    # Nhân hệ số attention với input x để áp dụng attention
    output = x * attention  # (b, c, h, w)
    return output, attention

# Ví dụ về mô hình sử dụng Attention Gate
class SimpleAttentionNet(nn.Module):
  def __init__(self):
    super(SimpleAttentionNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    # Điều chỉnh số lượng kênh trong AttentionGate sao cho phù hợp
    self.att_gate = AttentionGate(128, 128, 64)  # Attention Gate giữa conv2 và conv2 (giữa các lớp giống nhau)
    self.fc = nn.Linear(128*32*32, 10)  # Số lớp phân loại (10 lớp)

  def forward(self, x):
    # Pass qua các lớp convolution
    x = F.relu(self.conv1(x))  # (b, 64, h, w)
    x = F.relu(self.conv2(x))  # (b, 128, h, w)

    # Áp dụng Attention Gate
    gating_signal = x  # Ở đây, gating_signal chính là output từ conv2
    x, attention_map = self.att_gate(x, gating_signal)  # Áp dụng Attention

    # Flatten cho lớp fully connected
    x = x.view(x.size(0), -1)  # (b, 128*32*32)
    x = self.fc(x)  # Output phân loại
    return x, attention_map  # Trả về cả attention map để trực quan hóa

# Tạo input giả định
input_image = torch.randn(1, 3, 32, 32)  # Một ảnh RGB 32x32

# Khởi tạo mô hình
model = SimpleAttentionNet()

# Forward pass
output, attention_map = model(input_image)

# In kết quả
print("Output shape: ", output.shape)
print("Attention map shape: ", attention_map.shape)
