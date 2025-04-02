# MatAgent
A generative framework for exploring inorganic crystalline materials with desired properties.

## Requirements
- Python 3.12
- Git LFS

## Installation
### Install PyTorch
First, install PyTorch. For example, with CUDA 12.4, you can install PyTorch as follows:
```
$ pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Install PyG
Install PyTorch Geometric and its dependencies:
```
$ pip install torch_geometric
$ pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### Intall other dependencies
Install all other required packages with:
```
$ pip install .
```

## Setup OpenAI API Key
Set your OpenAI API Key as an environment variable:
```
$ export OPENAI_API_KEY="YOUR_API_KEY"
```

## Running the code
### Running the inference script
After installation, run the inference script:
```
$ matagent-inference --use_planning --data_path "./data/mp_20/train.csv" --n_init 1 --n_iterations 16 --target_value -3.8
```
Here, the `--data_path` parameter should be set to the path containing data used for sampling initial compositions.
### Initialize with Retriever
To initialize composition with Retriever, set the `--initial_guess` parameter to 'retriever'.
```
$ matagent-inference --use_planning --initial_guess "retriever" --data_path "./data/mp_20/train.csv" --n_init 1 --n_iterations 16 --target_value -3.8
```
### Generate with additional constraints
To impose additional constraints, use the `--additional_prompt` parameter.
```
$ matagent-inference --use_planning --data_path "./data/mp_20/train.csv" --n_init 1 --n_iterations 16 --target_value -3.8 --additional_prompt "ADDITIONAL PROMPT"
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

## References
This project was primarily built upon [CDVAE](https://github.com/txie-93/cdvae), [DiffCSP](https://github.com/jiaor17/DiffCSP), [ComFormer](https://github.com/divelab/AIRS/tree/main/OpenMat/ComFormer), and MatExpert[MatExpert](https://github.com/BangLab-UdeM-Mila/MatExpert).
```
@article{xie2021crystal,
  title={Crystal Diffusion Variational Autoencoder for Periodic Material Generation},
  author={Xie, Tian and Fu, Xiang and Ganea, Octavian-Eugen and Barzilay, Regina and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2110.06197},
  year={2021}
}
```
```
@article{jiao2024crystal,
  title={Crystal Structure Prediction by Joint Equivariant Diffusion}, 
  author={Rui Jiao and Wenbing Huang and Peijia Lin and Jiaqi Han and Pin Chen and Yutong Lu and Yang Liu},
  journal={arXiv preprint arXiv:2309.04475},
  year={2023},
}
```
```
@article{yan2024complete,
  title={Complete and Efficient Graph Transformers for Crystal Material Property Prediction}, 
  author={Keqiang Yan and Cong Fu and Xiaofeng Qian and Xiaoning Qian and Shuiwang Ji},
  journal={arXiv preprint arXiv:2403.11857}
  year={2024},
}
```
```
@article{ding2024matexpert,
  title={MatExpert: Decomposing Materials Discovery by Mimicking Human Experts}, 
  author={Qianggang Ding and Santiago Miret and Bang Liu}, 
  journal={arXiv preprint arXiv:2410.21317}
  year={2024},
}
```