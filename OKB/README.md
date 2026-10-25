# Option Keyboard Basis (OKB)

Code for reproducing the experiments in the paper "Constructing an Optimal Behavior Basis for the Option Keyboard".

## Running Experiments

Install the required dependencies via pip in the root directory of the repository:

```bash
pip install -e .
```

### Minecart

python experiments/okb_minecart.py --g=20 --use_ok=True --weight_selection=okb --num_nets=5 --run_name="OKB"

### HighWay

python experiments/okb_highway.py --use_ok=True --weight_selection=okb --run_name="OKB"

### FetchPickAndPlace

python experiments/okb_pickplace.py --use_ok=True --weight_selection=okb --num_nets=10 --g=10 --d=8 --run_name="OKB"

### Item Collection

For training the OKB agent, use the following command:

python experiments/okb_item_collector.py --use_ok=True --weight_selection=okb --run_name="OKB"

For evaluating the OKB in the non-linear task, use the following command:

python experiments/okb_item_collector.py --run_name="OKB Sequential Reward Task" --load_dir="PATH_TO_SAVED_AGENT" --single_task=sequential
