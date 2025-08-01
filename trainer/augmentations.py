import torch
import torch.nn.functional as F
import numpy as np

def add_noise(x, noise_factor=0.2):
    """
    Add random Gaussian noise to EEG data (torch tensor).
    x: Shape [b, channels, time_steps] (torch tensor)
    noise_factor: Scale of the Gaussian noise
    """
    # Generate Gaussian noise with the same shape as x
    noise = torch.randn_like(x) * noise_factor  # 使用 torch.randn_like 来生成和 x 形状相同的噪声
    
    # Add noise to the input data
    return x + noise

def add_noise_to_time_segment(x, noise_factor=0.2, segment_length=64):
    """
    Add random Gaussian noise to a random time segment in EEG data.
    x: Shape [b, channels, time_steps] (torch tensor)
    noise_factor: Scale of the Gaussian noise
    segment_length: Length of the time segment where noise will be added
    """
    b, channels, time_steps = x.shape
    
    # 随机选择一个时间片段的起始点
    start_time = torch.randint(0, time_steps - segment_length, (b,))  # 在时间维度上随机选择起点
    end_time = start_time + segment_length  # 计算结束时间点
    
    # 为选择的时间片段生成噪声
    noise = torch.randn(b, channels, segment_length).to(x.device) * noise_factor
    
    # 将噪声添加到对应时间片段
    for i in range(b):
        x[i, :, start_time[i]:end_time[i]] += noise[i]
    
    return x

def random_channel_swap(data, num_channels_to_swap=8):
    """
    对EEG样本进行随机多个通道的交换，输入为形状为 (b, channels, time_points) 的Tensor
    :param data: 输入的EEG数据，形状为 (b, channels, time_points)
    :param num_channels_to_swap: 需要交换的通道数，默认为2
    :return: 随机交换后的EEG数据
    """
    b, c, t = data.shape
    
    # 确保选择的通道数小于等于总通道数
    if num_channels_to_swap > c:
        raise ValueError(f"num_channels_to_swap should be less than or equal to the total number of channels ({c})")

    # 随机选择多个通道
    chan_idx = torch.randperm(c)[:num_channels_to_swap]  # 随机选择多个通道的索引
    #print(chan_idx)
    data_copy = torch.randn(b, c, t)
    data_copy = data.clone()

    # 将选定的通道之间的数据进行交换
    for i in range(1, num_channels_to_swap):
        data[:, i, :] = data_copy[:, chan_idx[i], :]
        data[:, chan_idx[i], :] = data_copy[:, i, :]

    return data

def frequency_domain_augmentation(x, freq_factor=0.2):
    """
    在频域上对脑电信号进行增强，避免显式使用for循环。
    x: Shape [b, channels, time_points] (torch tensor)
    freq_factor: 调整频率的系数，控制增强的强度。
    """
    b, channels, time_points = x.shape
    
    # 对信号进行 FFT
    x_freq = torch.fft.fft(x, dim=-1)
    
    # 获取频率域的幅度和相位
    magnitude = torch.abs(x_freq)
    phase = torch.angle(x_freq)
    
    # 随机噪声生成
    noise = torch.randn_like(magnitude) * freq_factor * magnitude
    
    # 调整幅度
    new_magnitude = magnitude + noise
    
    # 生成新的频域信号
    new_x_freq = new_magnitude * torch.exp(1j * phase)
    
    # 逆 FFT 转换回时域
    x_freq_augmented = torch.fft.ifft(new_x_freq, dim=-1).real
    
    return x_freq_augmented


def mixup_augmentation(data, label, alpha=0.2, num_classes=2):
    """ 
    Apply MixUp augmentation on the given data and labels.
    
    data: (b, channels, time_points) tensor
    label: (b,) tensor of labels for each sample (assumed to be class indices)
    alpha: Mixup coefficient to control the linear combination of two samples
    num_classes: Number of classes (only needed if labels are to be one-hot encoded)
    
    Returns:
        mixed_data: Mixed data tensor (b, channels, time_points)
        mixed_label: Mixed labels tensor
    """
    # Sample lambda from Beta distribution
    lambda_ = torch.distributions.Beta(alpha, alpha).sample([data.size(0)]).to(data.device)  # Shape (b,)

    # Randomly shuffle the data and labels
    indices = torch.randperm(data.size(0)).to(data.device)  # Shuffle indices

    # Mix the data
    mixed_data = lambda_.view(-1, 1, 1) * data + (1 - lambda_.view(-1, 1, 1)) * data[indices]

    # Mix the labels
    mixed_label = lambda_.view(-1, 1) * label + (1 - lambda_.view(-1, 1)) * label[indices]
    
    return mixed_data, mixed_label


def data_aug(data, label, method):
    if method == "an":
        data = add_noise(data)
    elif method == "antts":
        data = add_noise_to_time_segment(data)
    elif method == "rcs":
        data = random_channel_swap(data)
    elif method == "fda":
        data = frequency_domain_augmentation(data)
    elif method == "ma":
        data, label = mixup_augmentation(data, label)

    return data, label