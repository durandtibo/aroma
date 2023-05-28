# MultiTHUMOS

## Information about the dataset

The MultiTHUMOS dataset contains dense, multilabel, frame-level action annotations for 30 hours
across 400 videos in the THUMOS'14 action detection dataset.
It consists of 38,690 annotations of 65 action classes, with an average of 1.5 labels per frame and
10.5 action classes per video.

- Project
  page: [http://ai.stanford.edu/~syyeung/everymoment.html](http://ai.stanford.edu/~syyeung/everymoment.html)
- Research paper: Yeung S., Russakovsky O., Jin N., Andriluka M., Mori G., Fei-Fei L.
  *Every Moment Counts: Dense Detailed Labeling of Actions in Complex Videos.*
  IJCV 2017 [link](http://arxiv.org/pdf/1507.05738)

## Download data

`eternity` provides a functionality to download annotation data.
The `download_annotations` function can be used to download automatically the annotation data.
The following example shows how to download the annotation data in the
path `/path/to/data/multithumos`.

```python
from pathlib import Path
from aroma.datasets.multithumos import download_annotations

path = Path("/path/to/data/multithumos")
download_annotations(path)
print(list(path.iterdir()))
```

The output should look like:

```textmate
[PosixPath('/path/to/data/multithumos/README'),
 PosixPath('/path/to/data/multithumos/annotations'),
 PosixPath('/path/to/data/multithumos/class_list.txt')]
```

Note that is possible to download the data manually by following the instructions from
the [project webpage](http://ai.stanford.edu/~syyeung/everymoment.html).

The remaining of the documentation assumes the data are stored in the
directory `/path/to/data/multithumos/`.

## Action prediction task

This section explains how to prepare the data for the action prediction task.

### Get the event data

After the data are downloaded, you can get the event sequences by using the `load_event_data`
function.
This function returns the data and metadata.
The following example shows how to get the all the event sequences:

```python
from pathlib import Path
from aroma.datasets.multithumos import load_event_data

data, metadata = load_event_data(Path("/path/to/data/multithumos/"))
print(data)
print(metadata)
```

The output should look like:

```textmate
BatchDict(
  (video_id) BatchList(batch_size=413)
  (action_index) BatchedTensorSeq(dtype=torch.int64, shape=torch.Size([413, 1235]), device=cpu, batch_dim=0, seq_dim=1)
  (start_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([413, 1235]), device=cpu, batch_dim=0, seq_dim=1)
  (end_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([413, 1235]), device=cpu, batch_dim=0, seq_dim=1)
)

{'action_vocab': Vocabulary(
  counter=Counter({'BaseballPitch': 1, 'BasketballBlock': 1, 'BasketballDribble': 1, ...}),
  index_to_token=('BaseballPitch', 'BasketballBlock', 'BasketballDribble', ...),
  token_to_index={'BaseballPitch': 0, 'BasketballBlock': 1, 'BasketballDribble': 2, ...},
)}
```

The MultiTHUMOS dataset has two official dataset splits: `val` and `test`.
The following example shows how to get only the event sequences from the validation split.

```python
from pathlib import Path
from aroma.datasets.multithumos import load_event_data

data, metadata = load_event_data(Path("/path/to/data/multithumos/"), split="val")
print(data)
print(metadata)
```

The output should look like:

```textmate
BatchDict(
  (video_id) BatchList(batch_size=200)
  (action_index) BatchedTensorSeq(dtype=torch.int64, shape=torch.Size([200, 622]), device=cpu, batch_dim=0, seq_dim=1)
  (start_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([200, 622]), device=cpu, batch_dim=0, seq_dim=1)
  (end_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([200, 622]), device=cpu, batch_dim=0, seq_dim=1)
)

{'action_vocab': Vocabulary(
  counter=Counter({'BaseballPitch': 1, 'BasketballBlock': 1, 'BasketballDribble': 1, ...}),
  index_to_token=('BaseballPitch', 'BasketballBlock', 'BasketballDribble', ...),
  token_to_index={'BaseballPitch': 0, 'BasketballBlock': 1, 'BasketballDribble': 2, ...},
)}
```

The validation split should contain 200 sequences and the maximum sequence length is 622.
The following example shows how to get only the event sequences from the test split.

```python
from pathlib import Path
from aroma.datasets.multithumos import load_event_data

data, metadata = load_event_data(Path("/path/to/data/multithumos/"), split="test")
print(data)
print(metadata)
```

The output should look like:

```textmate
BatchDict(
  (video_id) BatchList(batch_size=213)
  (action_index) BatchedTensorSeq(dtype=torch.int64, shape=torch.Size([213, 1235]), device=cpu, batch_dim=0, seq_dim=1)
  (start_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([213, 1235]), device=cpu, batch_dim=0, seq_dim=1)
  (end_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([213, 1235]), device=cpu, batch_dim=0, seq_dim=1)
)

{'action_vocab': Vocabulary(
  counter=Counter({'BaseballPitch': 1, 'BasketballBlock': 1, 'BasketballDribble': 1, ...}),
  index_to_token=('BaseballPitch', 'BasketballBlock', 'BasketballDribble', ...),
  token_to_index={'BaseballPitch': 0, 'BasketballBlock': 1, 'BasketballDribble': 2, ...},
)}
```

The test split should contain 213 sequences and the maximum sequence length is 1235.
