# Evaluation Command Reference

The `App/` directory contains the untrusted SGX application sources and the five maintained Python evaluators. The former batch sweep shell scripts have been removed because they only wrapped repeated evaluator invocations.

Run the commands below from:

```bash
cd sgx-private-ann/App
```

## 1. Build the required implementation

Build exactly one implementation before running its evaluator:

```bash
make -C .. VARIANT=ours -j
make -C .. VARIANT=ohnsw -j
make -C .. VARIANT=compass_tee -j
```

The selected implementation is recorded in:

```bash
cat ../bin/ACTIVE_VARIANT
```

The evaluator commands use the compatibility paths `../bin/app` and `../bin/enclave.signed.so`. Rebuilding another variant updates those paths, so always verify `ACTIVE_VARIANT` before collecting results.

To inspect all evaluator options:

```bash
python3 eval_deferred_rebuild.py --help
python3 eval_deferred_rebuild_new.py --help
python3 eval_deferred_rebuild_with_gt_ivecs_mrr_fixed.py --help
python3 eval_deferred_rebuild_with_gt_ivecs_ohnsw_mrr_fixed.py --help
python3 eval_deferred_rebuild_with_gt_ivecs_compass_tee_mrr_fixed.py --help
```

## 2. SIFT1M

The following examples assume that the SIFT1M split index is stored under `../Application/offline_split_output`.

All documented SIFT1M runs use `query_start=0`, `num_queries=100`, and `topk=10`.

### 2.1 Ours without PQ

```bash
make -C .. VARIANT=ours -j

python3 eval_deferred_rebuild.py \
  --data-root ../../data/sift1m/sift \
  --meta-json ../Application/offline_split_output/meta.json \
  --server-index ../Application/offline_split_output/server_index.bin \
  --encrypted-server-index ../Application/offline_split_output/server_layers \
  --client-cache ../Application/offline_split_output/client_cache.bin \
  --app-bin ../bin/app \
  --signed-enclave ../bin/enclave.signed.so \
  --query-start 0 \
  --num-queries 100 \
  --topk 10 \
  --T0 5 \
  --T1 1 \
  --w 10 \
  --save results/sift1m_ours_no_pq_T0_5_T1_1_w10_q100.json
```

### 2.2 Ours with PQ hints

```bash
make -C .. VARIANT=ours -j

python3 eval_deferred_rebuild.py \
  --data-root ../../data/sift1m/sift \
  --meta-json ../Application/offline_split_output/meta.json \
  --server-index ../Application/offline_split_output/server_index.bin \
  --encrypted-server-index ../Application/offline_split_output/server_layers \
  --client-cache ../Application/offline_split_output/client_cache.bin \
  --app-bin ../bin/app \
  --signed-enclave ../bin/enclave.signed.so \
  --query-start 0 \
  --num-queries 100 \
  --topk 10 \
  --T0 5 \
  --T1 1 \
  --w 10 \
  --pq-codebook ../Application/offline_split_output/pq_hints/pq_codebook.bin \
  --pq-hint-table ../Application/offline_split_output/pq_hints/pq_hint_table.bin \
  --pq-efn0 24 \
  --pq-hint-linear-threshold 8192 \
  --save results/sift1m_ours_pq_T0_5_T1_1_w10_efn24_q100.json
```

The last maintained Ours sweep used these parameter ranges:

| Experiment | Parameters |
|---|---|
| Fixed comparison | `T0=5`, `T1=1`, `w=10`, 100 queries, with and without PQ |
| No-PQ sweep | `T0 in {4,5,6}`, `T1=1`, `w in {8,10,12}`, 100 queries |
| PQ sweep | `T0=5`, `T1=1`, `w=10`, `pq_efn0 in {16,24,32}`, hint threshold in `{8192,16384,32768}`, 100 queries |

### 2.3 OHNSW

```bash
make -C .. VARIANT=ohnsw -j

python3 eval_deferred_rebuild_new.py \
  --search-mode ohnsw \
  --ohnsw-tau 32 \
  --data-root ../../data/sift1m/sift \
  --meta-json ../Application/offline_split_output/meta.json \
  --server-index ../Application/offline_split_output/server_index.bin \
  --encrypted-server-index ../Application/offline_split_output/server_layers \
  --client-cache ../Application/offline_split_output/client_cache.bin \
  --app-bin ../bin/app \
  --signed-enclave ../bin/enclave.signed.so \
  --query-start 0 \
  --num-queries 100 \
  --topk 10 \
  --T0 6 \
  --T1 1 \
  --w 10 \
  --save results/sift1m_ohnsw_T0_6_T1_1_w10_tau32_q100.json
```

For the former dense sweep, keep `T0=6`, vary `T1` over `{1,2}`, vary `w` over `{10,12}`, and vary `tau` from 16 to 80. The densest points used increments of four or eight. Additional full-sweep points used `T1=3` and `w=16`.

### 2.4 Compass-in-TEE

```bash
make -C .. VARIANT=compass_tee -j

python3 eval_deferred_rebuild_new.py \
  --search-mode compass-tee \
  --compass-ef 20 \
  --compass-efn0 12 \
  --compass-efn1 0 \
  --compass-efspec 1 \
  --data-root ../../data/sift1m/sift \
  --meta-json ../Application/offline_split_output/meta.json \
  --server-index ../Application/offline_split_output/server_index.bin \
  --encrypted-server-index ../Application/offline_split_output/server_layers \
  --client-cache ../Application/offline_split_output/client_cache.bin \
  --app-bin ../bin/app \
  --signed-enclave ../bin/enclave.signed.so \
  --query-start 0 \
  --num-queries 100 \
  --topk 10 \
  --T0 1 \
  --T1 4 \
  --w 20 \
  --pq-codebook ../Application/offline_split_output/pq_hints/pq_codebook.bin \
  --pq-hint-table ../Application/offline_split_output/pq_hints/pq_hint_table.bin \
  --pq-efn0 12 \
  --pq-hint-linear-threshold 8192 \
  --save results/sift1m_compass_tee_T1_4_ef20_efn0_12_q100.json
```

The last Compass-in-TEE SIFT1M sweep used `T1 in {4,8}`, `ef in {10,20,30,40,50}`, `efn0 in {8,12,16,24}`, `efn1=0`, `efspec=1`, and a hint threshold of 8192. Set `w` to the same value as `compass-ef`.

## 3. MS MARCO 1M

The following examples assume that the MS MARCO subset and split index are stored under the paths used by the latest experiments.

Common evaluation inputs are:

```text
Data root:       ../../data/msmarco/mrr_subset_1m
ANN groundtruth: ../../data/msmarco/sift/msmarco_1000k_groundtruth.ivecs
Qrels:           ../../data/msmarco/mrr_subset_1m/qrels_subset.tsv
Query IDs:       ../../data/msmarco/mrr_subset_1m/query_ids.tsv
Offline index:   ../Application/offline_split_output_msmarco_oram_1000k
```

All maintained MS MARCO commands use `query_start=0`, `num_queries=100`, and `topk=10` unless a different query count is explicitly required.

### 3.1 Ours with PQ hints

```bash
make -C .. clean
make -C .. VARIANT=ours SGX_HNSW_MAX_VECTOR_DIM=768 -j

python3 eval_deferred_rebuild_with_gt_ivecs_mrr_fixed.py \
  --data-root ../../data/msmarco/mrr_subset_1m \
  --gt-ivecs ../../data/msmarco/sift/msmarco_1000k_groundtruth.ivecs \
  --mrr-qrels ../../data/msmarco/mrr_subset_1m/qrels_subset.tsv \
  --mrr-query-ids ../../data/msmarco/mrr_subset_1m/query_ids.tsv \
  --meta-json ../Application/offline_split_output_msmarco_oram_1000k/meta.json \
  --server-index ../Application/offline_split_output_msmarco_oram_1000k/server_index.bin \
  --encrypted-server-index ../Application/offline_split_output_msmarco_oram_1000k/server_layers \
  --client-cache ../Application/offline_split_output_msmarco_oram_1000k/client_cache.bin \
  --app-bin ../bin/app \
  --signed-enclave ../bin/enclave.signed.so \
  --query-start 0 \
  --num-queries 100 \
  --topk 10 \
  --T0 8 \
  --T1 1 \
  --w 16 \
  --pq-codebook ../Application/offline_split_output_msmarco_oram_1000k/pq_hints/pq_codebook.bin \
  --pq-hint-table ../Application/offline_split_output_msmarco_oram_1000k/pq_hints/pq_hint_table.bin \
  --pq-efn0 32 \
  --pq-hint-linear-threshold 8192 \
  --save results/msmarco_ours_T0_8_T1_1_w16_efn32_q100.json
```

The last Ours MS MARCO sweep used `T0 in {6,8,10}`, `T1=1`, `w in {8,16}`, `pq_efn0 in {24,32,48}`, and a hint threshold of 8192.

### 3.2 OHNSW

```bash
make -C .. clean
make -C .. VARIANT=ohnsw SGX_HNSW_MAX_VECTOR_DIM=768 -j

python3 eval_deferred_rebuild_with_gt_ivecs_ohnsw_mrr_fixed.py \
  --search-mode ohnsw \
  --ohnsw-tau 32 \
  --data-root ../../data/msmarco/mrr_subset_1m \
  --gt-ivecs ../../data/msmarco/sift/msmarco_1000k_groundtruth.ivecs \
  --mrr-qrels ../../data/msmarco/mrr_subset_1m/qrels_subset.tsv \
  --mrr-query-ids ../../data/msmarco/mrr_subset_1m/query_ids.tsv \
  --meta-json ../Application/offline_split_output_msmarco_oram_1000k/meta.json \
  --server-index ../Application/offline_split_output_msmarco_oram_1000k/server_index.bin \
  --encrypted-server-index ../Application/offline_split_output_msmarco_oram_1000k/server_layers \
  --client-cache ../Application/offline_split_output_msmarco_oram_1000k/client_cache.bin \
  --app-bin ../bin/app \
  --signed-enclave ../bin/enclave.signed.so \
  --query-start 0 \
  --num-queries 100 \
  --topk 10 \
  --T0 6 \
  --T1 1 \
  --w 10 \
  --save results/msmarco_ohnsw_T0_6_T1_1_w10_tau32_q100.json
```

The last OHNSW MS MARCO sweep used the following `(T0,T1,w,tau)` configurations:

```text
(6,1,8,8)   (6,1,8,16)  (6,1,8,24)  (6,1,8,32)  (6,1,8,48)
(6,1,9,20)  (6,1,9,28)  (6,1,9,36)
(6,1,10,16) (6,1,10,24) (6,1,10,32) (6,1,10,40)
(6,1,12,32) (6,1,12,48) (6,2,10,32)
```

### 3.3 Compass-in-TEE

```bash
make -C .. clean
make -C .. VARIANT=compass_tee SGX_HNSW_MAX_VECTOR_DIM=768 -j

python3 eval_deferred_rebuild_with_gt_ivecs_compass_tee_mrr_fixed.py \
  --search-mode compass-tee \
  --compass-ef 20 \
  --compass-efn0 12 \
  --compass-efn1 0 \
  --compass-efspec 1 \
  --data-root ../../data/msmarco/mrr_subset_1m \
  --gt-ivecs ../../data/msmarco/sift/msmarco_1000k_groundtruth.ivecs \
  --mrr-qrels ../../data/msmarco/mrr_subset_1m/qrels_subset.tsv \
  --mrr-query-ids ../../data/msmarco/mrr_subset_1m/query_ids.tsv \
  --meta-json ../Application/offline_split_output_msmarco_oram_1000k/meta.json \
  --server-index ../Application/offline_split_output_msmarco_oram_1000k/server_index.bin \
  --encrypted-server-index ../Application/offline_split_output_msmarco_oram_1000k/server_layers \
  --client-cache ../Application/offline_split_output_msmarco_oram_1000k/client_cache.bin \
  --app-bin ../bin/app \
  --signed-enclave ../bin/enclave.signed.so \
  --query-start 0 \
  --num-queries 100 \
  --topk 10 \
  --T0 1 \
  --T1 4 \
  --w 20 \
  --pq-codebook ../Application/offline_split_output_msmarco_oram_1000k/pq_hints/pq_codebook.bin \
  --pq-hint-table ../Application/offline_split_output_msmarco_oram_1000k/pq_hints/pq_hint_table.bin \
  --pq-efn0 12 \
  --pq-hint-linear-threshold 8192 \
  --save results/msmarco_compass_tee_T1_4_ef20_efn0_12_q100.json
```

The last Compass-in-TEE MS MARCO sweep used `T1 in {4,8}`, `ef in {10,20,30,40,50}`, `efn0 in {8,12,16,24}`, `efn1=0`, `efspec=1`, and a hint threshold of 8192. Set `w` to the same value as `compass-ef`.

## 4. Maintained evaluators

```text
eval_deferred_rebuild.py
eval_deferred_rebuild_new.py
eval_deferred_rebuild_with_gt_ivecs_mrr_fixed.py
eval_deferred_rebuild_with_gt_ivecs_ohnsw_mrr_fixed.py
eval_deferred_rebuild_with_gt_ivecs_compass_tee_mrr_fixed.py
```

Offline index construction, PQ generation, and dataset preparation utilities remain in `../Application/`.
