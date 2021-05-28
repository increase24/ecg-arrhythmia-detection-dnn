from torch.utils.data.dataset import Dataset
import numpy as np
import scipy.io as sio
import json
import tqdm
import os
from scipy.signal import resample

class CINC17(Dataset):
    def __init__(self, data_json, step):
        super(CINC17, self).__init__()
        self.STEP = step
        self.ecgs, self.labels = self._load_dataset(data_json, self.STEP) # -> list, list
        self.classes = sorted(set(l for label in self.labels for l in label)) # ['A', 'N', 'O', '~']
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}
        self.mean, self.std = self._compute_mean_std(self.ecgs)
        num_examples = len(self.ecgs)
        examples = zip(self.ecgs, self.labels)
        examples = sorted(examples, key = lambda x: x[0].shape[0])

    def _load_dataset(self, data_json, step):
        with open(data_json, 'r') as fid:
            data = [json.loads(l) for l in fid]
        labels = []; ecgs = []
        for d in tqdm.tqdm(data):
            labels.append(d['labels'])
            ecgs.append(resample(self._load_ecg(d['ecg'], step), 8960))  # resample to 8960
        return ecgs, labels   

    def _load_ecg(self, record, step):
        if os.path.splitext(record)[1] == ".npy":
            ecg = np.load(record)
        elif os.path.splitext(record)[1] == ".mat":
            ecg = sio.loadmat(record)['val'].squeeze()
        else: # Assumes binary 16 bit integers
            with open(record, 'r') as fid:
                ecg = np.fromfile(fid, dtype=np.int16)

        trunc_samp = step * int(len(ecg) // step)
        return ecg[:trunc_samp]

    def _compute_mean_std(self, x):
        x = np.hstack(x)
        return (np.mean(x).astype(np.float32),
            np.std(x).astype(np.float32))
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ecg = self.ecgs[idx]
        ecg = (ecg - self.mean) / self.std
        ecg = np.expand_dims(ecg, axis = 0)
        label = self.class_to_int[self.labels[idx]]
        return ecg.astype('float32'), label

if __name__ == "__main__":
    data_json = "experiments/cinc17/train.json"
    trainset = CINC17(data_json, 256)
    import matplotlib.pyplot as plt
    ecg, label = trainset[2000]
    ecg_clip = ecg[0, :4000]
    x = np.linspace(0, len(ecg_clip), len(ecg_clip), endpoint=False)
    y = ecg_clip
    fig = plt.figure()
    plt.plot(x, y, '-')
    plt.legend(['ecg'], loc='best')
    int2arrhythm = {0:'Normal', 1:'AF', 2:'Other rhythm', 3:'Noisy'}
    plt.title(int2arrhythm[label])
    #plt.show()
    fig.savefig('./figs/ecg_signal.png', dpi=300, bbox_inches='tight')

    
