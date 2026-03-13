
# Create a Class that takes the text and using a tokenizer gives input and target tensors
# Create a DataLoader Class that returns batches of Data for Training
import torch
from torch.utils.data import DataLoader,Dataset
import tiktoken 

class createDatasetTensors(Dataset):
    def __init__(self,text,context_window,stride,tokenizer):
        self.input = []
        self.target = []

        tokens = tokenizer.encode(text)
        l = len(tokens)

        for i in range(0,l-context_window,stride):
            ip = tokens[i:i+context_window]
            op = tokens[i+1:i+context_window+1]

            self.input.append(torch.tensor(ip))
            self.target.append(torch.tensor(op))
    def __len__(self):
         return len(self.input)
    
    def __getitem__(self,idx):
        return self.input[idx],self.target[idx]

def create_dataloader(context_window,stride,model="gpt2",batch_size=32,shuffle=True,drop_last=True,num_workers = 0):

        tokenizer = tiktoken.get_encoding(model)
        with open('data/verdict.txt','r') as f:
            text = f.read()
        
        dataset = createDatasetTensors(text,context_window,stride,tokenizer)
        print(len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle = shuffle,
            drop_last=drop_last,
            num_workers = num_workers
        )
        print(len(dataloader))
        return dataloader,dataset







