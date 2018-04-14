from torch.utils.data import Dataset

# data loader
class gtzandata(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __getitem__(self,index):

        # todo : random cropping audio to 3-second
        #start = random.randint(0,self.x[index].shape[1] - num_frames)
        #mel = self.x[index][:,start:start+num_frames]
        mel = self.x[index]

        entry = {'mel': mel, 'label': self.y[index]}

        return entry

    def __len__(self):
        return self.x.shape[0]