import h5py
import torch
import numpy as np
from models.inverse import inference, get_model
from tools import processer_canny

def load_pytorch_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    
def sliding_window(data, window_size, step_size):
    for i in range(0, len(data) - window_size + 1, step_size):
        yield data[i:i + window_size]

def pad_tensor(tensor, window_size):
    pad_size = window_size - 1
    return torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, pad_size))

def process_frames(frames, model, processor, window_size=16, step_size=1):
    # 首先对所有帧应用processor
    processed_frames = torch.stack([processor(f, 2) for f in frames], dim=0).cuda()
    # 然后进行填充
    padded_frames = pad_tensor(processed_frames, window_size)
    results = []
    
    for window in sliding_window(padded_frames, window_size, step_size):
        window_result = inference(model, window, [""])[0]  # 只取第一个结果
        results.append(window_result)
    
    return np.array(results)

def process_group(group, model, processor):
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            process_group(group[key], model, processor)
        elif isinstance(group[key], h5py.Dataset) and key == 'frames':
            frames = group[key][:]
            results = process_frames(frames, model, processor)

            if 'action' in group:
                del group['action']
            group.create_dataset('action', data=results, compression="gzip", chunks=True)
            
            print(f"处理完成: {group.name}")

def process_hdf5(file_path, model, processor):
    with h5py.File(file_path, 'r+') as f:
        process_group(f, model, processor)

    print(f"所有处理完成，结果已写入 {file_path}")

    
# 使用示例
if __name__ == "__main__":
    file_path = "/home/casia/workspace/zero.hdf5"
    processer = processer_canny()
    model = get_model().cuda()  # 这里应该是你的模型实例
    load_pytorch_model(model, "/home/casia/workspace/BC/output/inverse_results/TorchTrainer_29cb0_00000_0_2024-09-11_20-41-27/checkpoint_000042/model.pt")
    process_hdf5(file_path, model, processer)