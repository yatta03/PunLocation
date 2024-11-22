## 更

PyTorch Version: 2.5.1  
CUDA Available: True  
CUDA Version: 12.1  
cudnn version: 90100  
python: 3.11

執行程式，供參考：  
colab: [test1](https://colab.research.google.com/drive/1zjA7UTHwZlfta1NipE5aC4hZoo-Qr0LW?usp=sharing)  
(result: all.log, pun_visualize.ipynb)

## Joint Detection and Location of English Puns

Code for the NAACL-19 paper: Joint Detection and Location of English Puns.
This paper proposes to jointly address pun detection and location tasks by a sequence labeling approach with a newly designed tagging scheme.

### Requirements

Python 3.6
Pytorch 0.4

### Word embedding

Download the pretrained word embeddings [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/). Put the file under the folder `embeddings/`.

### Reproducing the experimental results

To reproduce the results, simply do the following command:

```
bash run.sh
```

### Cite

```
@InProceedings{zou-19-joint,
  author    = {Zou, Yanyan and Lu, Wei},
  title     = {Joint Detection and Location of English Puns},
  booktitle = {Proceedings of NAACL},
  year={2019}
}
```

```

## Contact

Yanyan Zou and Wei Lu, Singapore University of Technology and Design

Please feel free to drop an email at yanyan_zou@mymail.sutd.edu.sg for questions.
```

This implementation is inspired by the [work](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling)
