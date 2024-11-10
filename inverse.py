import h5py
import torch
import numpy as np
from models.inverse import inference, get_model
from tools import processer_canny, discrete_to_continuous

def load_pytorch_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    
def sliding_window(data, window_size, step_size):
    for i in range(0, len(data) - window_size + 1, step_size):
        yield data[i:i + window_size]

def pad_tensor(tensor, window_size):
    pad_size = window_size - 1
    return torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, pad_size))

def process_frames(frames, model, processor, window_size=12, step_size=1):
    processed_frames = torch.stack([processor(f, 2) for f in frames], dim=0).cuda()
    padded_frames = pad_tensor(processed_frames, window_size)
    results = []
    
    for window in sliding_window(padded_frames, window_size, step_size):
        window_result = inference(model, window, [""])[0]  # Only take the first result
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
            group.create_dataset('action', data=results[:len(frames)], compression="gzip", chunks=True)
            
            print(f"Finish: {group.name}")

def process_hdf5(file_path, model, processor):
    with h5py.File(file_path, 'r+') as f:
        process_group(f, model, processor)

    print(f"Write to {file_path}")

    
if __name__ == "__main__":
    file_path = "/path/to/hdf5"
    processer = processer_canny()
    model = get_model().cuda()
    load_pytorch_model(model, "/path/to/pytorch_model.bin")
    process_hdf5(file_path, model, processer)
