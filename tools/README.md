# Modified Tools for Benchmarking

This directory contains the modified source code of the tools used in our experiments.  

---

## Prerequisites

- Each tool has different environment and library requirements, and where its provided README is incomplete, we provide the necessary instructions below.
- **It is necessary to set up a separate virtual environment** for each tool.
- Tools Invalidator and FIXCHECK require Defects4J to be set-up and added to path (Defects4J also requires Java 11). All relevant instructions can be found here: https://github.com/rjust/defects4j

## Results Files

All tools save results to a csv file. To generate the figures in the report with these csvs, they need to be moved to the `results` directory under the relevant tool subdirectory. Tool subdirectories have their github repository names which may not match with the way they are referenced in the report. The mapping between tool name and directory name is provided below for each tool. Once csvs are moved to the appropriate directory, they must have the same name e.g. `8h_deduplicated.csv`.

---

Below are instructions to replicate our benchmarking results.

## MIPI

- **Publication:**  
  Q.-N. Phung, M. Kim, and E. Lee,  
  *"Identifying incorrect patches in program repair based on meaning of source code"*  
  *IEEE Access*, vol. 10, pp. 12012–12030, 2022.

- **Type:** Static
- **Directory:** `mipi`
- **Original Repository:** https://github.com/ngocpq/MIPI

**Instructions:**  
1. Follow setup instructions in `mipi/README.md` to setup environment and download model
2. Change your working directory to `src/mipi-code2vec` and run `pip install -r requirements.txt` and `bash build_extractor.sh`
3. We found a missing dependency: `pip install protobuf==3.20.*`
4. To run experiments, ensure your working directory is `src/mipi-code2vec` and run `python mipi_sample.py mipi_patches_8h_deduplicated.json`
5. Results will be saved under `src/mipi-code2vec/mipi_patches_8h_deduplicated_results.csv`

---

## Yang et al.

- **Publication:**  
A. Z. Yang, S. Kolak, V. J. Hellendoorn, R. Martins, and C. L. Goues,
*“Revisiting unnaturalness for automated program repair in the era of large language models”*
*arXiv* preprint arXiv:2404.15236, 2024.

- **Type:** Static
- **Directory:** `entropy-apr-replication`
- **Original Repository:** https://github.com/squaresLab/entropy-apr-replication

**Instructions:**  
1. As outlined in the tool's README, `CUDA version >= 11.4` along with the necessary drivers is required, as well as `torch (version 2.0.1)`.
2. Run `bash init_env.sh`
3. We found a missing dependency: `pip install accelerate`
4. To generate entropy scores for each patch, run `python patches/fyp_patch_entropy.py`
5. To use the entropy scores to classify each patch, run `python analysis_notebooks/analysis_fyp.py`
6. Results will be saved under `entropy_analysis_fyp_cutoff_-0.55.csv`

---

## FIXCHECK

- **Publication:**  
F. Molina, J. M. Copia, and A. Gorla,
*“Improving patch correctness analysis via random testing and large language models”*
in 2024 *IEEE Conference on Software Testing, Verification and Validation (ICST)*, pp. 317–328, 2024.

- **Type:** Dynamic
- **Directory:** `fixcheck`
- **Original Repository:** https://github.com/facumolina/fixcheck 

**Instructions:**  
1. Follow setup instructions in `fixcheck/README.md`
2. We found an issue setting up llama-cpp which we resolved by installing via URL: `pip install --no-cache-dir llama-cpp-python==0.2.85 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122`
3. In `experiments/run-fixcheck-fyp.py` replace the four constants at the top as instructed by the comments to include your own java path etc.
4. In one terminal, launch the LLM by running `python llms/replit-code.py` and wait a few seconds until it is running.
5. In another terminal, launch fixcheck by running `python experiments/run-fixcheck-fyp.py`
6. Once fixcheck is done, get patch classifications by running `python get_results.py`
7. Results will be saved under `fixcheck_results_8h_test.csv`

---

## Invalidator

- **Publication:**  
T. Le-Cong, D.-M. Luong, X. B. D. Le, D. Lo, N.-H. Tran, B. Quang-Huy, and Q.-T. Huynh,
*“Invalidator: Automated patch correctness assessment via semantic and syntactic reasoning”*
*IEEE Transactions on Software Engineering*, vol. 49, no. 6, pp. 3411–3429, 2023.

- **Type:** Dynamic
- **Directory:** `Invalidator`
- **Original Repository:** https://github.com/thanhlecongg/Invalidator 

**Instructions:**  
1. The Invalidator replication package did not provide details of Daikon setup. This is the procedure we used to install it:
  - `wget https://plse.cs.washington.edu/daikon/download/daikon-5.8.20.tar.gz`
  - `tar -xzf daikon-5.8.20.tar.gz`
  - Then add Daikon relevant values to bashrc: `export DAIKONDIR="/home/user/workspace/daikon-5.8.20"` `export PATH=$DAIKONDIR:$PATH` `export CLASSPATH=$DAIKONDIR/daikon.jar:$CLASSPATH`
  - `make -C $DAIKONDIR rebuild-everything`
2. Install required libraries: `conda env create -f environment.yml`
3. In `fyp.py` at the top replace the JAVA_11_HOME constant with your path to your java 11 installation and the patches path as instructed by the comments.
4. Generate invariants using Daikon and Defects4J by running `python fyp.py`
5. To analyse invariants and classify patches, run `python experiment.py --c 0`
6. Results will be saved under `results.csv`

---

## LLM4PatchCorrect

- **Publication:**  
X. Zhou, B. Xu, K. Kim, D. Han, H. H. Nguyen, T. Le-Cong, J. He, B. Le, and D. Lo,
*“Leveraging large language model for automatic patch correctness assessment”*
*IEEE Transactions on Software Engineering*, vol. 50, no. 11, pp. 2865–2883, 2024.

- **Type:** Learning
- **Directory:** `LLM4PatchCorrectness`
- **Original Repository:** https://github.com/Xin-Zhou-smu/LLM4PatchCorrectness 

**Instructions:**  
1. Run `bash install_library.sh`
2. To download the model via terminal:
  - `pip install gdown`
  - `gdown --folder https://drive.google.com/drive/folders/1MryWp2iqXAVo4UHxnN-bTspQkysM7Fpy?usp=sharing`
  - Place it under `pretrained_model/best`
3. We used deepspeed as our GPU was not selected by default: `pip install deepspeed`
4. To start the tool, run `bash run_pipeline.sh`
5. To generate classifications from LLM4PatchCorrect run `python read_results_enhanced.py`
6. Results will be saved under `8h-deduplicated_results.csv`

---

## Tian et al.

- **Publication:**  
H. Tian, K. Liu, A. K. Kabor´e, A. Koyuncu, L. Li, J. Klein, and T. F. Bissyand´e
*“Evaluating representation learning of code changes for predicting patch correctness in program repair”*
in Proceedings of the 35th *IEEE/ACM International Conference on Automated Software Engineering, ASE* ’20, (New York, NY, USA), p. 981–992, Association for Computing Machinery, 2021.

- **Type:** Learning
- **Directory:** `DL4PatchCorrectness`
- **Original Repository:** https://github.com/TruX-DTF/DL4PatchCorrectness 

**Instructions:**
1. Setup the server environment: 
  - `mkdir server && cd server`
  - `wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip`
  - `unzip wwm_cased_L-24_H-1024_A-16.zip`
  - The server requires its own environment with these packages: `pip install tensorflow==1.14 && pip install bert-serving-client==1.10.0 && pip install bert-serving-server==1.10.0 && pip install protobuf==3.20.1`
  - Run: `bert-serving-start -model_dir ./wwm_cased_L-24_H-1024_A-16 -num_worker=1 -max_seq_len=360` NOTE this needs to be in a separate terminal, set num_worker to the number of gpus available otherwise the server will fail to start.
2. Setup the tool environment:
  - `pip install bert-serving-client==1.10.0 && pip install scikit-learn && pip install gensim && pip install nltk`
  - `python -m nltk.downloader punkt`
3. To classify patches, first ensure the server is running, then change your working directory to `DL4PatchCorrectness/prediction` and run `python fyp.py`
4. Results will be saved under `patch_results_dt.csv`

---