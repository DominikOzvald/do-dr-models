import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os

SPECIAL_SYMBOLS = ["<PAD>", "<UNK>", "<SOS>", "\n"]


def extract_tag(log):
    tags = ["<FLAKY>", "<DRIFT>", "<SECURITY>", "<SILENT>"]
    tag_num = 0
    for i, tag in enumerate(tags):
        if log.startswith(tag):
            log = log[len(tag) :]
            tag_num = i + 1
    return log, tag_num


def extract_tagged(file_name, max_in_len=200):
    tagged_pairs = []
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                log = line.split(" ", maxsplit=1)
                if len(log) > 1:
                    log, tag = extract_tag(log[1])
                    tagged_pairs.append((log[:max_in_len], tag))

    except Exception as e:
        print("Skipping file:", file_name, e)
    return tagged_pairs


class CharVocab:
    def __init__(self):
        self.str2int = {}

        for i, c in enumerate(SPECIAL_SYMBOLS):
            self.str2int[c] = i

        for i in range(32, 127):
            self.str2int[chr(i)] = i - 32 + len(SPECIAL_SYMBOLS)
        self.int2str = dict((v, k) for k, v in self.str2int.items())

    def encode(self, text):
        encoded = []
        for c in text:
            if c in self.str2int:
                encoded.append(self.str2int[c])
            else:
                encoded.append(self.str2int["<UNK>"])
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, indexes):
        decoded = ""
        for index in indexes:
            index = int(index)
            if index in self.int2str:
                decoded += self.int2str[index]
            else:
                decoded += self.int2str[0]
        return decoded

    def __len__(self):
        return len(self.str2int)


class DummyLogDataSet(Dataset):
    def __init__(
        self,
        log_dir: str = None,
        step: int = 5,
        frame_size: int = 30,
        max_in_len: int = 200,
        pad_tag: int = 0,
    ):
        super().__init__()
        self.step = step
        self.frame_size = frame_size
        self.vocab = CharVocab()
        self.data = []
        self.files = []
        self.max_in_len = max_in_len
        self.pad_tag = pad_tag
        if log_dir:
            log_files = [file for file in os.listdir(log_dir) if file[-4:] == ".txt"]
            for log_file in log_files:
                self.add_from_file(os.path.join(log_dir, log_file))

    def add_from_file(self, file_name):
        pairs = extract_tagged(file_name, self.max_in_len)
        file_start = len(self.data)
        num_frames = 0
        for i in range(0, len(pairs), self.step):
            self.data.append(pairs[i : i + self.frame_size])
            num_frames += 1
        self.files.append((file_name, file_start, num_frames))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        logs, tags = zip(*self.data[item])
        with torch.no_grad():
            lengths = torch.Tensor([len(log) for log in logs])
            tags = torch.Tensor(tags)
            enc_logs = [self.vocab.encode(log) for log in logs]
            padded_log = [
                F.pad(log, (0, self.max_in_len - log.size(0)), value=0)
                for log in enc_logs
            ]
            frame = torch.stack(padded_log, dim=0)
            frame_len = frame.size(0)
            if frame_len < self.frame_size:
                frame = F.pad(frame, (0, 0, 0, self.frame_size - frame_len), value=0)
                lengths = torch.cat([lengths, torch.ones(self.frame_size - frame_len)])
                tags = torch.cat(
                    [tags, self.pad_tag * torch.ones(self.frame_size - frame_len)]
                )
                mask = torch.cat(
                    [torch.zeros(frame_len), torch.ones(self.frame_size - frame_len)]
                )
            else:
                mask = torch.zeros(frame_len)
            tags = tags.to(torch.long)
        return frame, lengths, mask, tags
    
    def get_str_log_item(self,item):
        logs, tags = zip(*self.data[item])
        return logs
