# A transition-based dependency parser for both projective and non-projective trees
This is a PyTorch implement of ['Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations'.](https://aclweb.org/anthology/Q16-1023) and ['Arc-Hybrid Non-Projective Dependency Parsing with a Static-Dynamic Oracle'](https://www.aclweb.org/anthology/W17-6314).

It is a transition-based dependency parser for both projective and non-projective trees, using BiLSTM networks and arc-hybrid systems.
It is just a naive implement of these paper without much optimization, so it does not show .

## Requirements
+ Python 3.6 (>=)
+ PyTorch 1.0 (>=)

## About dataset
The dataset I used for training is a treebank in Chinese. It is only for class use, so I can't upload it to github.
It is preprocessed into 'json' format. Each word in the sentence has these attributes: 'id', 'word', 'pos', 'father', 'emb'.

## Reference
+ https://github.com/elikip/bist-parser
+ https://github.com/UppsalaNLP/uuparser
+ https://github.com/MathijsMul/lm_parser/blob/master/arc_hybrid.py
