# Drug-SCAGE (Based on SCAGE)

此專案主要源自原始 SCAGE 專案，並在其基礎上做了實務訓練流程調整與擴充（例如 3-step MPP finetune 流程）。

- Original project: [KazeDog/SCAGE](https://github.com/KazeDog/scage)
- This repository: customized workflow for molecular property prediction (MPP)

## 專案重點

- 保留 SCAGE 的核心模型與資料流程。
- 新增/調整多個訓練腳本，包含：
  - `3_finetune_2step_mpp.py`
  - `3_finetune_3step_mpp.py`（目前主要流程）
  - `3.1_finetune_search_aggregate.py`（彙整搜尋結果）
- 支援以 JSON/YAML 搜尋設定做超參數網格搜尋。

## 3-step MPP Finetune 流程

`3_finetune_3step_mpp.py` 的流程如下：

1. **Stage 1**: 凍結 encoder，只訓練 `head_Graph`。  
2. **Stage 2**: 載入 Stage 1 最佳模型，維持 encoder 凍結，訓練三個 head（graph / finger / atom_fg）。  
3. **Stage 3**: 載入 Stage 2 最佳模型，解凍全模型進行 finetune。  
4. 以 valid split 主指標挑選最佳候選，並輸出最終權重與分數報告。  

## 目錄概覽

- `models/`：SCAGE 模型與 layers
- `data_process/`：資料處理與切分工具
- `datasets/`：dataloader
- `utils/`：訓練、評估、報表等工具
- `config/`：訓練與搜尋設定
- `weights/`：預訓練/微調權重（本 repo 預設忽略追蹤）

## 環境與執行（Docker）

建議使用 docker 執行（GPU）：

```bash
docker run --gpus all --name scage -itd -p 8888:8888 -v "$(pwd):/workspace" scage:cu121
docker exec -it scage bash
cd /workspace
```

## 快速開始

### 1) 3-step finetune（建議）

```bash
python 3_finetune_3step_mpp.py --search_config ./config/caco2_finetune_search_3step.yaml
```

### 2) 2-step finetune（相容舊流程）

```bash
python 3_finetune_2step_mpp.py --search_config ./config/caco2_finetune_search.yaml
```

### 3) 聚合搜尋結果

```bash
python 3.1_finetune_search_aggregate.py
```

## 常用輸出

- `finetune_result/`：各 trial/stage 訓練紀錄與 checkpoint
- `outputs/`：預測結果與分數彙整
- `result/`：額外結果輸出資料夾

## Git 版本控制建議

本專案建議忽略大型資料與訓練產物，`.gitignore` 建議至少包含：

- `data/`
- `weights/`
- `outputs/`
- `result/`
- `finetune_result/`

## 致謝與授權

本專案基於 SCAGE 進行延伸開發，感謝原作者與貢獻者。

- Upstream: [KazeDog/SCAGE](https://github.com/KazeDog/scage)
- 請同時遵循原始專案授權條款（見 `LICENSE`）。
