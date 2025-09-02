# MatAgent
A generative framework for interpretable and targeted inorganic materials design using diffusion-based generation, property prediction, and LLM-driven reasoning.

## Installation
### Prepare environment
```
pip install uv
uv venv .venv --python 3.12 
source .venv/bin/activate
```
### Install PyTorch
First, install PyTorch. For example, with CUDA 12.4, you can install PyTorch as follows:
```
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Install PyG
Install PyTorch Geometric and its dependencies:
```
uv pip install torch_geometric
uv pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### Intall other dependencies
Install all other required packages with:
```
uv pip install -e .
```

## Setup OpenAI API Key
Set your OpenAI API Key as an environment variable:
```
export OPENAI_API_KEY="YOUR_API_KEY"
```

## Running the code
### Running the inference script
After installation, run the inference script:
```
matagent-inference --use_planning --data_path "./data/mp_20/train.csv" --n_init 1 --n_iterations 16 --target_value -3.8
```
Here, the command parameters control the execution as follows:
- `--use_planning`: Use tool-assisted Planning and Proposition
- `--data_path`: Path to the dataset used for sampling initial compositions
- `--n_init`: Number of independent initializations to perform
- `--n_iterations`: Number of iterations for each independent run
- `--target_value`: Target formation energy (in eV/atom)


Additional configurable parameters are available in agent4crys/scripts/inference.py.
### Generate with additional constraints
To impose additional constraints, use the `--additional_prompt` parameter.
```
$ matagent-inference --use_planning --data_path "./data/mp_20/train.csv" --n_init 1 --n_iterations 16 --target_value -3.8 --additional_prompt "ADDITIONAL PROMPT"
```

### Initialize with Retriever
#### Checkpoint download instructions
- To initialize using the Retriever method, first download the model checkpoint from Hugging Face. Use the following command to download the checkpoint:
    ```
    $ wget https://huggingface.co/izumitkh/matagent-retriever/resolve/main/best_model.pth
    ```
- Move `best_model.pth` to the `agent4crys/component/contriever/pretrain` directory.

#### Generate with Retriever initialization 
After placing the checkpoint in the correct location, you can execute generation by setting the `--initial_guess` parameter to "retriever".
```
$ matagent-inference --use_planning --initial_guess "retriever" --data_path "./data/mp_20/train.csv" --n_init 1 --n_iterations 16 --target_value -3.8
```

## Citation
```
@article{takahara2025accelerated,
  title={Accelerated Inorganic Materials Design with Generative AI Agents}, 
  author={Izumi Takahara and Teruyasu Mizoguchi and Bang Liu},
  journal={arXiv preprint arXiv:2504.00741},
  year={2025},
}
```

## Acknowledgement
This project was primarily built upon [CDVAE](https://github.com/txie-93/cdvae), [DiffCSP](https://github.com/jiaor17/DiffCSP), [ComFormer](https://github.com/divelab/AIRS/tree/main/OpenMat/ComFormer), and [MatExpert](https://github.com/BangLab-UdeM-Mila/MatExpert).