# CD-T: Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition
This is a mechanistic interpretation tool to build circuits of internal components (e.g. attention heads) in transformers. Our code supports both BERT-like models and GPT-like models.

## Organization
- `greater_than_task`: utilities for the Greater-than task.
- `im_utils`: utilities for the docstring task.
- `methods`: utilities for the pathology report classification of the BERT model that we tested. (the pathology report eperiment on BERT is not included in the paper and the data/model is also not uploaded to this repo because of protected patient information. however, the provided CD-T implementation for BERT is general. readers should feel free to plug in their own BERTs and play with the code.)
- `notebooks/correctness_tests`: sanity checks of our contextual decomposition implementation.
- `notebooks`: major circuit experiments demonstrated on GPT2 a shown in the paper. we additionally include an `local_importance.ipynb` to compare and contrast CD-T with other modern local importance tools, such as LIME, SHAP, and layer intergrated gradients.
- `pyfunctions`: general utilities.

## Set up environment
```bash
pip install -r requirements.txt
```

## Citation
If you use any of our code in your work, please cite:
```bash
@misc{hsu2024efficientautomatedcircuitdiscovery,
      title={Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition}, 
      author={Aliyah R. Hsu and Georgia Zhou and Yeshwanth Cherapanamjeri and Yaxuan Huang and Anobel Y. Odisho and Peter R. Carroll and Bin Yu},
      year={2024},
      eprint={2407.00886},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.00886}, 
}
```
