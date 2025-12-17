# Local_Inn

## Setup
Prerequisites
- Python 3.11 or higher

```
source setup/setup.bash
```

Add your rosbag to `data/`

## Commands

Dataloader (Convert from rosbag)
```
python dataloader.py <EXP_NAME> <ROSBAG_NAME>
```

Preprocessing
```
python dataprocess.py <EXP_NAME>
```

Training
```
python model.py <EXP_NAME>
```

Analysis
```
python analyse_results <EXP_NAME>
```
## Results
![Localisation Results](/images/error_map.png)


Author's Implementation - https://github.com/zzangupenn/Local_INN
