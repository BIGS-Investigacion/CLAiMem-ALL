CLAM Installation Guide <img src="clam-logo.png" width="350px" align="right" />
===========
Next, use the environment configuration file to create a conda environment:
```shell
conda env create -f env.yml
```

Activate the environment:
```shell
conda activate clam_latest
```

Copy '.env example' and rename as '.env'. Include your HF_TOKEN to download models from Hugging Face repository.

When done running experiments, to deactivate the environment:
```shell
conda deactivate clam_latest
```
Please report any issues in the public forum.

[Return to main page.](README.md)
