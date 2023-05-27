# Breakfast

## Information about the dataset

This dataset contains 10 actions related to breakfast preparation, performed by 52 different
individuals in 18
different kitchens.
Overall, ∼77 hours of video (> 4 million frames) are recorded.
The cameras used were webcams, standard industry cameras (Prosilica GE680C) as well as a stereo
camera (BumbleBee ,
Pointgrey, Inc).
To balance out viewpoints, we also mirrored videos recorded from laterally-positioned cameras.
To reduce the overall amount of data, all videos were down-sampled to a resolution of 320×240 pixels
with a frame rate
of 15 fps.
The number of cameras used varied from location to location (n = 3 − 5).
The cameras were uncalibrated and the position of the cameras changes based on the location.

- Project
  webpage: [https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/)
- Research paper:
  Kuehne, Arslan, and Serre.
  *The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities.*
  CVPR 2014.

## Download data

The annotation data are stored in a Google Drive. You have to follow the instructions in
the [Download section](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/#Downloads)
to download the
data.

The annotation data used for the action prediction task are in the `segmentation_coarse.tar.gz`
and ` segmentation_fine.tar.gz` files. Then, you have to extract the files.
It is possible to use the function `download_annotations` to automatically download the data:

```pycon
>>> from pathlib import Path
>>> from aroma.datasets.breakfast import download_annotations
>>> download_annotations(Path("/path/to/data/breakfast/"))
>>> list(path.iterdir())
[PosixPath('/path/to/data/breakfast/segmentation_coarse'),
 PosixPath('/path/to/data/breakfast/segmentation_fine')]
```

The remaining of the documentation assumes the data are stored in the
directory `/path/to/data/breakfast/`.

## Action prediction task

This section explains how to prepare the data for the action prediction task.

### Get the event data

After the data are downloaded, you can get the event sequences by using the `load_event_data`
function.
This function returns the data and metadata.
The following example shows how to get the event sequences by using the coarse annotations:

```python
from pathlib import Path
from aroma.datasets.breakfast import load_event_data

data, metadata = load_event_data(Path("/path/to/data/breakfast/segmentation_coarse"))
print(data)
print(metadata)
```

The output should look like:

```textmate
BatchDict(
  (action_index) tensor([[ 0, 16, 21,  ..., -1, -1, -1],
            [ 0, 21,  1,  ..., -1, -1, -1],
            [ 0, 16, 21,  ..., -1, -1, -1],
            ...,
            [ 0, 25, 23,  ..., -1, -1, -1],
            [ 0, 13, 23,  ..., -1, -1, -1],
            [ 0, 13, 23,  ..., -1, -1, -1]], batch_dim=0, seq_dim=1)
  (cooking_activity) BatchList(data=['cereals', 'cereals', 'cereals', ..., 'tea'])
  (end_time) tensor([[[ 30.],
             [150.],
             [428.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 55.],
             [233.],
             [405.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 13.],
             [246.],
             [624.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            ...,

            [[448.],
             [558.],
             [754.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 11.],
             [101.],
             [316.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 37.],
             [ 92.],
             [229.],
             ...,
             [ nan],
             [ nan],
             [ nan]]], batch_dim=0, seq_dim=1)
  (person_id) BatchList(data=['P03', 'P04', 'P05', ..., 'P54'])
  (start_time) tensor([[[  1.],
             [ 31.],
             [151.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 56.],
             [234.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 14.],
             [247.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            ...,

            [[  1.],
             [449.],
             [559.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 12.],
             [102.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 38.],
             [ 93.],
             ...,
             [ nan],
             [ nan],
             [ nan]]], batch_dim=0, seq_dim=1)
)
{'action_vocab': Vocabulary(
  counter=Counter({'SIL': 1016, 'pour_milk': 199, 'cut_fruit': 176, ..., 'stir_tea': 2}),
  index_to_token=('SIL', 'pour_milk', 'cut_fruit', 'crack_egg', ..., 'stir_tea'),
  token_to_index={'SIL': 0, 'pour_milk': 1, 'cut_fruit': 2, ..., 'stir_tea': 47},
)}
```

It is also possible to use the function `fetch_event_data`, which automatically download the data if
they are missing:

```python
from pathlib import Path
from aroma.datasets.breakfast import fetch_event_data

data, metadata = fetch_event_data(
    Path("/path/to/data/breakfast/"), name="segmentation_coarse"
)
```

By default, the duplicate event sequences are removed.
There are duplicate event sequences because there are multiple videos of the same scene.
You can set `remove_duplicate=False` to keep the duplicate event sequences.

```python
from pathlib import Path
from aroma.datasets.breakfast import load_event_data

data, metadata = load_event_data(
    Path("/path/to/data/breakfast/segmentation_coarse/"), remove_duplicate=False
)
print(data)
print(metadata)
```

The output should look like:

```textmate
BatchDict(
  (action_index) tensor([[ 0, 16, 18,  ..., -1, -1, -1],
            [ 0, 16, 18,  ..., -1, -1, -1],
            [ 0, 16, 18,  ..., -1, -1, -1],
            ...,
            [ 0, 13, 20,  ..., -1, -1, -1],
            [ 0, 13, 20,  ..., -1, -1, -1],
            [ 0, 13, 20,  ..., -1, -1, -1]], batch_dim=0, seq_dim=1)
  (cooking_activity) BatchList(data=['cereals', 'cereals', 'cereals', ..., 'tea'])
  (end_time) tensor([[[ 30.],
             [150.],
             [428.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 30.],
             [150.],
             [428.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 30.],
             [150.],
             [428.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            ...,

            [[ 37.],
             [ 92.],
             [229.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 37.],
             [ 92.],
             [229.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 37.],
             [ 92.],
             [229.],
             ...,
             [ nan],
             [ nan],
             [ nan]]], batch_dim=0, seq_dim=1)
  (person_id) BatchList(data=['P03', 'P03', 'P03', ..., 'P54'])
  (start_time) tensor([[[  1.],
             [ 31.],
             [151.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 31.],
             [151.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 31.],
             [151.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            ...,

            [[  1.],
             [ 38.],
             [ 93.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 38.],
             [ 93.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 38.],
             [ 93.],
             ...,
             [ nan],
             [ nan],
             [ nan]]], batch_dim=0, seq_dim=1)
)
{'action_vocab': Vocabulary(
  counter=Counter({'SIL': 1016, 'pour_milk': 199, 'cut_fruit': 176, ..., 'stir_tea': 2}),
  index_to_token=('SIL', 'pour_milk', 'cut_fruit', 'crack_egg', ..., 'stir_tea'),
  token_to_index={'SIL': 0, 'pour_milk': 1, 'cut_fruit': 2, ..., 'stir_tea': 47},
)}
```

It is possible to use the same function to get the event sequences with the fine annotations.

```python
from pathlib import Path
from aroma.datasets.breakfast import load_event_data

data, metadata = load_event_data(Path("/path/to/data/breakfast/segmentation_fine/"))
print(data)
print(metadata)
```

The output should look like:

```textmate
BatchDict(
  (action_index) tensor([[  0,  44,  49,  ...,  -1,  -1,  -1],
            [  0,  83,  91,  ...,  -1,  -1,  -1],
            [  0,  44,  49,  ...,  -1,  -1,  -1],
            ...,
            [  0, 117,  62,  ...,  -1,  -1,  -1],
            [  0,  27,   1,  ...,  -1,  -1,  -1],
            [  0, 117, 115,  ...,  -1,  -1,  -1]], batch_dim=0, seq_dim=1)
  (cooking_activity) BatchList(data=['cereals', 'cereals', 'cereals', ..., 'tea'])
  (end_time) tensor([[[53.],
             [63.],
             [80.],
             ...,
             [nan],
             [nan],
             [nan]],

            [[45.],
             [47.],
             [92.],
             ...,
             [nan],
             [nan],
             [nan]],

            [[17.],
             [32.],
             [45.],
             ...,
             [nan],
             [nan],
             [nan]],

            ...,

            [[ 7.],
             [22.],
             [41.],
             ...,
             [nan],
             [nan],
             [nan]],

            [[52.],
             [63.],
             [92.],
             ...,
             [nan],
             [nan],
             [nan]],

            [[26.],
             [47.],
             [85.],
             ...,
             [nan],
             [nan],
             [nan]]], batch_dim=0, seq_dim=1)
  (person_id) BatchList(data=['P03', 'P04', 'P05', ..., 'P46'])
  (start_time) tensor([[[ 1.],
             [54.],
             [64.],
             ...,
             [nan],
             [nan],
             [nan]],

            [[ 1.],
             [46.],
             [48.],
             ...,
             [nan],
             [nan],
             [nan]],

            [[ 1.],
             [18.],
             [33.],
             ...,
             [nan],
             [nan],
             [nan]],

            ...,

            [[ 1.],
             [ 8.],
             [23.],
             ...,
             [nan],
             [nan],
             [nan]],

            [[ 1.],
             [53.],
             [64.],
             ...,
             [nan],
             [nan],
             [nan]],

            [[ 1.],
             [27.],
             [48.],
             ...,
             [nan],
             [nan],
             [nan]]], batch_dim=0, seq_dim=1)
)
{'action_vocab': Vocabulary(
  counter=Counter({'garbage': 774, 'move': 649, 'carry_knife': 403, ..., 'carry_capSalt': 3}),
  index_to_token=('garbage', 'move', 'carry_knife', ..., 'carry_capSalt'),
  token_to_index={'garbage': 0, 'move': 1, 'carry_knife': 2, ..., 'carry_capSalt': 177},
)}
```

### Filter the data by dataset split

The `load_event_data` function returns all the event sequence of the Breakfast dataset.
A common operation is to separate the data by dataset splits.
You can split the data by using the `filter_batch_by_dataset_split` function.
The following example shows how to filter the data for the `train1` split.

```python
from pathlib import Path
from aroma.datasets.breakfast import filter_batch_by_dataset_split, load_event_data

data, metadata = load_event_data(Path("/path/to/data/breakfast/segmentation_coarse/"))
data_train = filter_batch_by_dataset_split(data, "train1")
print(data_train)
```

The output should look like:

```textmate
BatchDict(
  (action_index) tensor([[ 0, 21,  1,  ..., -1, -1, -1],
            [ 0, 21,  1,  ..., -1, -1, -1],
            [ 0, 21,  1,  ..., -1, -1, -1],
            ...,
            [ 0, 25, 23,  ..., -1, -1, -1],
            [ 0, 13, 23,  ..., -1, -1, -1],
            [ 0, 13, 23,  ..., -1, -1, -1]], batch_dim=0, seq_dim=1)
  (cooking_activity) BatchList(data=['cereals', 'cereals', 'cereals', ..., 'tea'])
  (end_time) tensor([[[  9.],
             [269.],
             [474.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 39.],
             [322.],
             [521.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 26.],
             [190.],
             [362.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            ...,

            [[448.],
             [558.],
             [754.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 11.],
             [101.],
             [316.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 37.],
             [ 92.],
             [229.],
             ...,
             [ nan],
             [ nan],
             [ nan]]], batch_dim=0, seq_dim=1)
  (person_id) BatchList(data=['P16', 'P17', 'P18', ..., 'P54'])
  (start_time) tensor([[[  1.],
             [ 10.],
             [270.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 40.],
             [323.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 27.],
             [191.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            ...,

            [[  1.],
             [449.],
             [559.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 12.],
             [102.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 38.],
             [ 93.],
             ...,
             [ nan],
             [ nan],
             [ nan]]], batch_dim=0, seq_dim=1)
)
```

Similarly, the following example shows how to filter the data for the `test1` split.

```python
from pathlib import Path
from aroma.datasets.breakfast import filter_batch_by_dataset_split, load_event_data

data, metadata = load_event_data(Path("/path/to/data/breakfast/segmentation_coarse/"))
data_test = filter_batch_by_dataset_split(data, "test1")
print(data_test)
```

The output should look like:

```textmate
BatchDict(
  (action_index) tensor([[ 0, 16, 21,  ..., -1, -1, -1],
            [ 0, 21,  1,  ..., -1, -1, -1],
            [ 0, 16, 21,  ..., -1, -1, -1],
            ...,
            [ 0, 25, 23,  ..., -1, -1, -1],
            [ 0, 25, 23,  ..., -1, -1, -1],
            [ 0, 25, 23,  ..., -1, -1, -1]], batch_dim=0, seq_dim=1)
  (cooking_activity) BatchList(data=['cereals', 'cereals', 'cereals', ..., 'tea'])
  (end_time) tensor([[[ 30.],
             [150.],
             [428.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 55.],
             [233.],
             [405.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 13.],
             [246.],
             [624.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            ...,

            [[ 48.],
             [168.],
             [323.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 96.],
             [173.],
             [391.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[ 10.],
             [110.],
             [295.],
             ...,
             [ nan],
             [ nan],
             [ nan]]], batch_dim=0, seq_dim=1)
  (person_id) BatchList(data=['P03', 'P04', 'P05', ..., 'P15'])
  (start_time) tensor([[[  1.],
             [ 31.],
             [151.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 56.],
             [234.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 14.],
             [247.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            ...,

            [[  1.],
             [ 49.],
             [169.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 97.],
             [174.],
             ...,
             [ nan],
             [ nan],
             [ nan]],

            [[  1.],
             [ 11.],
             [111.],
             ...,
             [ nan],
             [ nan],
             [ nan]]], batch_dim=0, seq_dim=1)
)
```
