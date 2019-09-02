# 人人都会机器翻译系列1 - seq2seq模型 :beers:
---

相关链接：
[github :cat:](https://github.com/Niutranser-Li/Seq2Seq_NMT)
**邮箱地址:pig:**(hello_xiaojian@126.com)
**QQ:penguin:**(779388649)
**欢迎通过各种方式随时交流**

---

## 模型发展简述 :books:
**seq2seq**,全称为Sequence to Sequence，即为传统的Encoder-Decoder模型，该技术为深度神经网络模型最为经典的案例，突破了输入序列大小固定的限制，使得经典深度学习模型在**机器翻译**、**人机交互**、**自动文摘**等领域得到了突破性的进展。

本着化繁为简的目的，我们使用[**PyTorch**](https://pytorch.org/)实现了一个简易的seq2seq机器翻译模型，模型只保留了最基础的**Encoder-编码器**、**Decoder-解码器**部分，去除掉了许多修饰成分（Attention、Dropout、batch等），方便理解最基础的Encoder、Decoder工作原理。

## Part-I（数据加载部分 - dataLoader.py）
加载数据是神经网络模型必不可少的一部分，因为所有的深度学习模型均为数据驱动的，若没有足够的数据支撑，模型很难学习到最优的参数，达到不错的效果。同时，每一个模型均需要生成对应的数据格式以满足模型训练的需求。所以，数据加载部分是至关重要的一部分。

如果你想了解dataLoader.py部分的原始代码，请登录github查看源代码。
并可以通过以下方式运行此程序：
```
python3 dataLoader.py
```

## Part-II (模型定义部分 - seq2seq.py)
该代码部分定义了整个程序最核心的部分，机器翻译模型（**encoder**&**decoder**）部分。如下图所示：
<img width="600" src="https://github.com/Niutranser-Li/Seq2Seq_NMT/blob/master/img/encoder-decoder.png">


## Part-III (模型训练部分 - train.py)
```
python3 train.py -h
usage: train.py [-h] --epoch_num EPOCH_NUM [--embedding_size EMBEDDING_SIZE]
                [--hidden_size HIDDEN_SIZE] [--model_path MODEL_PATH]
                [--srcLang SRCLANG] [--tgtLang TGTLANG]
optional arguments:
  -h, --help            show this help message and exit
  --epoch_num EPOCH_NUM
                        Number of epoch to train.
  --embedding_size EMBEDDING_SIZE
                        Word Embedding Vector dimension size, default = 300
  --hidden_size HIDDEN_SIZE
                        Hidden size of RNN. default = 300
  --model_path MODEL_PATH
                        The path of encoder and decoder models.
  --srcLang SRCLANG     The language of source.
  --tgtLang TGTLANG     The language of target.
```
**使用**
```
CUDA_VISIBLE_DEVICES=3 python3 train.py --epoch_num 1 --embedding_size 300 --hidden_size 300
```

## Part-IV (模型测试部分)
```
python3 evaluate.py -h
usage: evaluate.py [-h] --encoder ENCODER --decoder DECODER
                   [--embedding_size EMBEDDING_SIZE]
                   [--hidden_size HIDDEN_SIZE] [--srcLang SRCLANG]
                   [--tgtLang TGTLANG]

optional arguments:
  -h, --help            show this help message and exit
  --encoder ENCODER     Encoder file path to load trained_encoder's learned
                        parameters.
  --decoder DECODER     Decoder file path to load trained_decoder's learned
                        parameters.
  --embedding_size EMBEDDING_SIZE
                        Word embedding vector dimension size. default = 300
  --hidden_size HIDDEN_SIZE
                        Hidden size of rnn. default = 300
  --srcLang SRCLANG     The language of source.
  --tgtLang TGTLANG     The language of target.
```
**使用**
```
CUDA_CISIBLE_DEVICES=3 python3 evaluate.py --encoder model/encoder.pth --decoder model/decoder.pth
```
