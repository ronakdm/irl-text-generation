import pickle
from torch.utils.data import Dataset


class COCOImageCaptionsDataset(Dataset):
    """COCO Image Captions dataset."""

    def __init__(self, pkl_file):
        """
        Parameters: pkl_file (string): Path to the file with tokenized sentences.
        """
        self.pkl_file = pkl_file
        self.truth, self.m_in, self.mask = pickle.load(
            open(pkl_file, "rb")
        )  # (num_sentences, m_in)
        self.truth = self.truth.cuda()
        self.m_in = self.m_in.cuda()
        self.mask = self.mask.cuda()

    def __len__(self):
        return len(self.truth)

    def __getitem__(self, idx):
        return self.truth[idx], self.m_in[idx], self.mask[idx]
