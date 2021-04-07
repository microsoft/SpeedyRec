# SpeedyRec

## Introduction
Pytorch Implention for [Training large-scale news recommenders with pretrained language models in the loop](https://arxiv.org/pdf/2102.09268.pdf)
## Requirements
```bash
pip install -r requirements.txt
```
## Data
See [example_data/README.md](example_data/README.md) for Dataset Format


## Usage
```python
python run_example.py --root_data_dir ./example_data/  --bus_connection True --content_refinement True --max_keyword_freq 100  --beta_for_cache 0.002 --max_step_in_cache 20  --mode train_test  --world_size 4 --pretrained_model_path None
```
More parameter information please refer to `src/parameter.py`


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
