
import torch
from asr.semantic_vad.models.model_BiLSTM import BiLSTM
from asr.semantic_vad.models.model_bert import Bert
from asr.semantic_vad.models.bert_model.tokenization import BertTokenizer
from asr.semantic_vad.models.bert_model.modeling import BertConfig

class SemanticComplete_BiLSTM():
    def __init__(self, model_path, vocab_path, device):
        self.device = device
        self.model = BiLSTM()
        params = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(params, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.vocab = {}
        with open(vocab_path) as f:
            for item in f.readlines():
                word = item.strip().split()[0]
                id = int(item.strip().split()[1])
                self.vocab[word] = id
        # warmup
        with torch.no_grad():
            logit = self.model(torch.LongTensor([0] * 128).unsqueeze(0).to(self.device))

    def map_value(self, input_value):
        if 0 <= input_value <= 0.4:
            return 1300 - 1000 * input_value
        elif 0.4 < input_value <= 0.8:
            return 1200 - 1000 * input_value
        elif 0.8 < input_value <= 1.0:
            return 1100 - 1000 * input_value
        else:
            return 500
           
    def forward(self, feature):
        ques = []
        for i in feature.strip():
            if i in self.vocab: 
                ques.append(self.vocab[i])
            else:
                ques.append(self.vocab['<unk>'])
        if len(ques) < 128:
            ques += [0] * (128-len(ques))
        with torch.no_grad():
            logit = self.model(torch.LongTensor(ques).unsqueeze(0).to(self.device))
        return int(self.map_value(logit.item()))


class SemanticComplete_Bert():
    def __init__(self, model_path, vocab_path, device):
        self.device = device
        self.hidden_size = 192
        self.pad_size = 64
        self.bert_config = BertConfig(vocab_size_or_config_json_file=21128, 
                                      hidden_size=self.hidden_size,
                                      num_hidden_layers=4, 
                                      num_attention_heads=4, 
                                      intermediate_size=384, 
                                      max_position_embeddings=128)
        self.tokenizer = BertTokenizer(vocab_file=vocab_path, max_len=self.pad_size)
        self.model = Bert(self.bert_config)
        params = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(params, strict=True)
        self.model.to(self.device)
        self.model.eval()
        # warmup
        with torch.no_grad():
            warmup_inputs = torch.LongTensor([2802]*3 + [0] * 61).unsqueeze(0).to(self.device)
            warmup_mask = torch.LongTensor([1]*3 + [0] * 61).unsqueeze(0).to(self.device)
            logit = self.model(warmup_inputs, warmup_mask)

    def map_value(self, input_value):
        if 0 <= input_value <= 0.4:
            return 1300 - 1000 * input_value
        elif 0.4 < input_value <= 0.8:
            return 1250 - 1000 * input_value
        elif 0.8 < input_value <= 1.0:
            return 1150 - 1000 * input_value
        else:
            return 500
           
    def forward(self, feature):
        token = self.tokenizer.tokenize(feature)
        token = ['[CLS]'] + token
        mask = []
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        seq_len = len(token)

        if seq_len < self.pad_size:
            mask = [1] * seq_len + [0] * (self.pad_size - seq_len)
            token_ids += ([0] * (self.pad_size - seq_len))
        else:
            mask = [1] * self.pad_size
            token_ids = token_ids[:self.pad_size]

        content = torch.LongTensor(token_ids).unsqueeze(0).to(self.device)
        mask = torch.LongTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(content, mask)
        return int(self.map_value(logit.item()))
    
