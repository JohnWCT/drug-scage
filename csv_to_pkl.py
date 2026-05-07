"""
csv_to_pkl.py - 將 CSV 檔案轉換為 evaluate_mpp.py 所需的 PKL 格式

此腳本包含後處理步驟，確保輸出 PKL 與現有 bace.pkl 格式一致。

使用方式:
=========
1. 轉換 CSV → PKL:
   python csv_to_pkl.py --taskname bace --dataroot ./data/mpp/raw/ --datatarget ./data/mpp/pkl_test/

2. 驗證新 PKL 與參考 PKL 是否一致:
   python csv_to_pkl.py --verify --new_pkl ./data/mpp/pkl_test/bace.pkl --ref_pkl ./data/mpp/pkl/bace.pkl

3. 轉換 + 驗證 (一步完成):
   python csv_to_pkl.py --taskname bace --dataroot ./data/mpp/raw/ --datatarget ./data/mpp/pkl_test/ \
       --verify --ref_pkl ./data/mpp/pkl/bace.pkl

4. 檢視 PKL 結構:
   python csv_to_pkl.py --inspect ./data/mpp/pkl/bace.pkl
"""

import argparse
import os
import pickle
import sys
import signal
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdchem
from tqdm import tqdm

# 確保專案根目錄在 sys.path 中
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _config import get_downstream_task_names
from data_process.compound_tools import (
    mol_to_data_pkl, CompoundKit, safe_index, get_dist_bar,
    rd_chem_enum_to_list
)


# ============================================================
# Bond 特徵提取工具
# ============================================================

# Bond vocab dict (與 CompoundKit 的 atom_vocab_dict 類似)
# 這些在舊版 mol_to_data_pkl 中有包含，但新版被移除
bond_vocab_dict = {
    'bond_dir': rd_chem_enum_to_list(rdchem.BondDir.values),
    'bond_type': rd_chem_enum_to_list(rdchem.BondType.values),
    'is_in_ring': [0, 1],
    'bond_stereo': rd_chem_enum_to_list(rdchem.BondStereo.values),
    'is_conjugated': [0, 1],
}

# dist_bar percentiles (from GlobalVar default)
DEFAULT_DIST_BAR = [25, 50, 75]


def extract_bond_features(mol, edges):
    """
    從 RDKit Mol 物件提取每條邊的 bond 特徵。
    
    這是舊版 mol_to_data_pkl 中包含但在當前版本中被移除的功能。
    包括: bond_dir, bond_type, is_in_ring, bond_stereo, is_conjugated
    
    設計思路:
    - 遍歷 mol.GetBonds()，按照 edges 中的 (i, j) 順序匹配
    - 如果找不到對應的 bond (自環)，使用預設值 0
    - 使用 safe_index 將枚舉值對應到 vocab dict 中的索引
    
    Args:
        mol: RDKit Mol 物件
        edges: numpy array, shape=(E, 2), 邊的端點索引
    
    Returns:
        dict: 包含 bond_dir, bond_type, is_in_ring, bond_stereo, is_conjugated 的 numpy arrays
    """
    bond_features = {name: [] for name in bond_vocab_dict.keys()}
    
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is None:
            # 自環或找不到 bond
            for name in bond_vocab_dict.keys():
                bond_features[name].append(0)
        else:
            bond_features['bond_dir'].append(
                safe_index(bond_vocab_dict['bond_dir'], bond.GetBondDir()))
            bond_features['bond_type'].append(
                safe_index(bond_vocab_dict['bond_type'], bond.GetBondType()))
            bond_features['is_in_ring'].append(
                safe_index(bond_vocab_dict['is_in_ring'], int(bond.IsInRing())))
            bond_features['bond_stereo'].append(
                safe_index(bond_vocab_dict['bond_stereo'], bond.GetStereo()))
            bond_features['is_conjugated'].append(
                safe_index(bond_vocab_dict['is_conjugated'], int(bond.GetIsConjugated())))
    
    for name in bond_vocab_dict.keys():
        bond_features[name] = np.array(bond_features[name], 'int64')
    
    return bond_features


def compute_edge_types(bond_type_arr, bond_dir_arr):
    """
    計算 edge_types: 結合 bond_type 和 bond_dir 的複合特徵。
    
    edge_type = bond_type * num_bond_dir + bond_dir
    
    Args:
        bond_type_arr: numpy array of bond type indices
        bond_dir_arr: numpy array of bond dir indices
    
    Returns:
        numpy array of edge type indices
    """
    num_bond_dir = len(bond_vocab_dict['bond_dir'])
    edge_types = bond_type_arr * num_bond_dir + bond_dir_arr
    return np.array(edge_types, 'int64')


def compute_bar_keys(data, dist_bar=None):
    """
    計算距離 percentile bar 特徵。
    
    這些在舊版 PKL 中存在，用於將距離矩陣壓縮為分位數統計:
    - atom_dist_bar: pair_distances 的 percentile
    - edge_dist_bar: edge_distances 的 percentile  
    - atom_bond_dist_bar: atom_bond_distances 的 percentile
    - spatial_pos_bar: spatial_pos 的 percentile
    
    Args:
        data: dict, 包含 pair_distances, edge_distances, atom_bond_distances, spatial_pos
        dist_bar: list of percentiles, 預設 [25, 50, 75]
    
    Returns:
        dict with added *_bar keys
    """
    if dist_bar is None:
        dist_bar = DEFAULT_DIST_BAR
    
    data['atom_dist_bar'] = get_dist_bar(data['pair_distances'], dist_bar)
    data['edge_dist_bar'] = get_dist_bar(data['edge_distances'], dist_bar)
    data['atom_bond_dist_bar'] = get_dist_bar(data['atom_bond_distances'], dist_bar)
    data['spatial_pos_bar'] = get_dist_bar(data['spatial_pos'], dist_bar)
    
    return data


# ============================================================
# 步驟 1: 讀取 CSV，提取 SMILES 與 Labels
# ============================================================
def load_finetune_dataset(input_path, target):
    """
    從 CSV 檔案讀取 SMILES 與對應的 label 欄位。
    
    邏輯:
    - 讀取 CSV，取出 'smiles' 欄位作為分子的 SMILES 表示
    - 根據 target (來自 task_configs) 取出對應的 label 欄位
    - 用 -1 填充 NaN 值 (代表缺失的 label)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[target]
    labels = labels.fillna(-1)
    assert len(smiles_list) == len(labels)
    return smiles_list, labels.values


# ============================================================
# 步驟 2: 將單一 SMILES 轉換為分子特徵 dict (含後處理)
# ============================================================
class _SmilesTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _SmilesTimeout("SMILES processing timed out")


def _process_smiles_with_label(args):
    """
    處理單一分子: SMILES → RDKit Mol → 分子圖特徵 dict → 後處理。
    
    邏輯:
    1. 用 RDKit 將 SMILES 解析為 Mol 物件
    2. 呼叫 mol_to_data_pkl(mol) 取得基本分子圖特徵
    3. 後處理: 
       a. 提取 bond 特徵 (bond_dir, bond_type, is_in_ring, bond_stereo, is_conjugated)
       b. 計算 edge_types
       c. 計算 *_bar 特徵
       d. 移除 bond_distances (舊版不包含)
    4. 附加 smiles 和 label
    """
    smiles, label, timeout_sec = args
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(max(1, int(timeout_sec)))
    try:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            reason = "parse_failed"
            print(f"[WARNING] 無法解析的 SMILES: {smiles}", file=sys.stderr)
            return {"ok": False, "smiles": smiles, "label": label, "reason": reason}
        
        # 核心轉換
        data = mol_to_data_pkl(mol)
        if data is None:
            reason = "feature_none"
            print(f"[WARNING] mol_to_data_pkl 返回 None: {smiles}", file=sys.stderr)
            return {"ok": False, "smiles": smiles, "label": label, "reason": reason}
        
        # 後處理: 提取 bond 特徵
        bond_feats = extract_bond_features(mol, data['edges'])
        data.update(bond_feats)
        
        # 後處理: 計算 edge_types
        data['edge_types'] = compute_edge_types(data['bond_type'], data['bond_dir'])
        
        # 後處理: 計算 *_bar 特徵
        data = compute_bar_keys(data)
        
        # 後處理: 移除舊版不包含的 key
        if 'bond_distances' in data:
            del data['bond_distances']
        
        # 附加 metadata
        data['smiles'] = smiles
        data['label'] = label
        
        return {"ok": True, "data": data}
    except _SmilesTimeout:
        reason = f"timeout_{timeout_sec}s"
        print(f"[WARNING] 單筆 SMILES 處理超時({timeout_sec}s)，已跳過: {smiles}", file=sys.stderr)
        return {"ok": False, "smiles": smiles, "label": label, "reason": reason}
    except Exception as e:
        reason = f"exception:{type(e).__name__}"
        print(f"[WARNING] 單筆 SMILES 處理失敗，已跳過: {smiles} | error={e}", file=sys.stderr)
        return {"ok": False, "smiles": smiles, "label": label, "reason": reason}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================
# 步驟 3: 批量轉換整個 CSV → PKL
# ============================================================
def process_csv_to_pkl(task_name, csv_dir, target_pkl_dir, num_cores=4, smiles_timeout_sec=60, failed_csv_path=None):
    """
    主要轉換函數: 讀取 CSV → 平行處理所有 SMILES → 輸出 PKL。
    """
    os.makedirs(target_pkl_dir, exist_ok=True)

    config = {
        'task_name': task_name,
        'path': os.path.join(csv_dir, f'{task_name}.csv')
    }
    config = get_downstream_task_names(config)
    
    print(f"=== 轉換設定 ===")
    print(f"  Task: {task_name}")
    print(f"  CSV: {config['path']}")
    print(f"  Target columns: {config['target']}")
    print(f"  Task type: {config['task']}")
    print(f"  Num cores: {num_cores}")
    print(f"  Per-SMILES timeout: {smiles_timeout_sec}s")
    
    smiles_list, labels = load_finetune_dataset(config['path'], config['target'])
    print(f"  Total samples: {len(smiles_list)}")
    
    args = [(s, l, smiles_timeout_sec) for s, l in zip(smiles_list, labels)]
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(_process_smiles_with_label, args), 
                           total=len(args), desc="Processing SMILES"))
    
    total_data = [r["data"] for r in results if r.get("ok")]
    failed_items = [r for r in results if not r.get("ok")]
    failed_count = len(failed_items)
    if failed_count > 0:
        print(f"  [WARNING] {failed_count} SMILES 無法解析，已跳過")
        if failed_csv_path:
            failed_df = pd.DataFrame(
                [{"smiles": r.get("smiles"), "label": r.get("label"), "reason": r.get("reason")}
                 for r in failed_items]
            )
            os.makedirs(os.path.dirname(failed_csv_path), exist_ok=True)
            failed_df.to_csv(failed_csv_path, index=False)
            print(f"  [INFO] 失敗樣本清單輸出: {failed_csv_path}")
    
    output_path = os.path.join(target_pkl_dir, f'{task_name}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(total_data, f)
    
    print(f"  輸出: {output_path}")
    print(f"  有效樣本數: {len(total_data)}")
    print(f"=== 轉換完成 ===\n")
    
    return output_path


# ============================================================
# 步驟 4: 驗證新 PKL 與參考 PKL 的一致性
# ============================================================
def verify_pkl(new_pkl_path, ref_pkl_path):
    """
    逐一比對新生成的 PKL 與參考 PKL 是否一致。
    """
    print(f"=== 驗證 PKL 一致性 ===")
    print(f"  新 PKL: {new_pkl_path}")
    print(f"  參考 PKL: {ref_pkl_path}")
    
    with open(new_pkl_path, 'rb') as f:
        new_data = pickle.load(f)
    with open(ref_pkl_path, 'rb') as f:
        ref_data = pickle.load(f)
    
    if len(new_data) != len(ref_data):
        print(f"  ✗ 樣本數量不一致: 新={len(new_data)}, 參考={len(ref_data)}")
        return False
    print(f"  ✓ 樣本數量一致: {len(new_data)}")
    
    mismatches = []
    for i in range(len(new_data)):
        new_item = new_data[i]
        ref_item = ref_data[i]
        
        new_keys = set(new_item.keys())
        ref_keys = set(ref_item.keys())
        if new_keys != ref_keys:
            mismatches.append({
                'index': i, 'type': 'keys_mismatch',
                'new_only': new_keys - ref_keys,
                'ref_only': ref_keys - new_keys
            })
            continue
        
        for key in ref_keys:
            new_val = new_item[key]
            ref_val = ref_item[key]
            
            if isinstance(ref_val, np.ndarray):
                if not isinstance(new_val, np.ndarray):
                    mismatches.append({
                        'index': i, 'key': key, 'type': 'type_mismatch',
                        'new_type': type(new_val).__name__, 'ref_type': 'ndarray'
                    })
                elif new_val.shape != ref_val.shape:
                    mismatches.append({
                        'index': i, 'key': key, 'type': 'shape_mismatch',
                        'new_shape': new_val.shape, 'ref_shape': ref_val.shape
                    })
                elif not np.array_equal(new_val, ref_val):
                    try:
                        max_diff = float(np.max(np.abs(new_val.astype(float) - ref_val.astype(float))))
                    except:
                        max_diff = -1
                    if max_diff >= 0 and max_diff < 1e-4:
                        mismatches.append({
                            'index': i, 'key': key, 'type': 'value_close',
                            'max_diff': max_diff
                        })
                    else:
                        mismatches.append({
                            'index': i, 'key': key, 'type': 'value_mismatch',
                            'max_diff': max_diff,
                            'new_sample': str(new_val.flatten()[:5]),
                            'ref_sample': str(ref_val.flatten()[:5])
                        })
            elif isinstance(ref_val, (dict, list)):
                if str(new_val) != str(ref_val):
                    mismatches.append({
                        'index': i, 'key': key, 'type': 'complex_mismatch',
                        'detail': f'new_len={len(new_val)}, ref_len={len(ref_val)}'
                    })
            else:
                if new_val != ref_val:
                    mismatches.append({
                        'index': i, 'key': key, 'type': 'scalar_mismatch',
                        'new_val': str(new_val)[:100],
                        'ref_val': str(ref_val)[:100]
                    })
    
    if not mismatches:
        print(f"  ✓ 所有 {len(new_data)} 個樣本完全一致!")
        print(f"=== 驗證通過 ===\n")
        return True
    else:
        print(f"  ✗ 發現 {len(mismatches)} 處不一致:")
        for m in mismatches[:20]:
            if m['type'] == 'keys_mismatch':
                print(f"    [樣本 {m['index']}] Keys 不一致 - 新增: {m['new_only']}, 缺少: {m['ref_only']}")
            elif m['type'] == 'type_mismatch':
                print(f"    [樣本 {m['index']}][{m['key']}] 類型不一致 - 新: {m['new_type']}, 參考: {m['ref_type']}")
            elif m['type'] == 'shape_mismatch':
                print(f"    [樣本 {m['index']}][{m['key']}] Shape 不一致 - 新: {m['new_shape']}, 參考: {m['ref_shape']}")
            elif m['type'] == 'value_close':
                print(f"    [樣本 {m['index']}][{m['key']}] 數值接近 (max_diff={m['max_diff']:.2e})")
            elif m['type'] == 'value_mismatch':
                print(f"    [樣本 {m['index']}][{m['key']}] 數值不一致 (max_diff={m['max_diff']:.2e})")
                print(f"      新: {m.get('new_sample','')}")
                print(f"      參考: {m.get('ref_sample','')}")
            elif m['type'] == 'scalar_mismatch':
                print(f"    [樣本 {m['index']}][{m['key']}] 值不一致 - 新: {m['new_val']}, 參考: {m['ref_val']}")
            elif m['type'] == 'complex_mismatch':
                print(f"    [樣本 {m['index']}][{m['key']}] 結構不一致 - {m['detail']}")
        
        if len(mismatches) > 20:
            print(f"    ... 還有 {len(mismatches) - 20} 處不一致未顯示")
        
        # 統計分佈
        key_counts = {}
        type_counts = {}
        for m in mismatches:
            k = m.get('key', 'KEYS')
            t = m['type']
            key_counts[k] = key_counts.get(k, 0) + 1
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"\n  不一致的 key 分佈:")
        for k, c in sorted(key_counts.items(), key=lambda x: -x[1]):
            print(f"    {k}: {c} 處")
        print(f"\n  不一致的類型分佈:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {t}: {c} 處")
        
        print(f"\n  [提示] 如果差異在 atom_pos/pair_distances 等 3D 座標相關欄位，")
        print(f"  可能是 RDKit conformer 生成的隨機性造成。")
        print(f"  若差異在 bond features，可能是 RDKit 版本差異。")
        print(f"=== 驗證失敗 ===\n")
        return False


# ============================================================
# 步驟 5: 顯示 PKL 結構 (用於偵錯)
# ============================================================
def inspect_pkl(pkl_path, num_samples=3):
    print(f"=== PKL 結構檢視: {pkl_path} ===")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  類型: {type(data).__name__}")
    print(f"  樣本數: {len(data)}")
    
    if len(data) > 0:
        sample = data[0]
        print(f"  Keys ({len(sample.keys())}): {sorted(sample.keys())}")
        for key in sorted(sample.keys()):
            val = sample[key]
            if isinstance(val, np.ndarray):
                print(f"    {key}: ndarray, shape={val.shape}, dtype={val.dtype}")
            elif isinstance(val, (dict, list)):
                print(f"    {key}: {type(val).__name__}, len={len(val)}")
            else:
                val_str = str(val)[:80]
                print(f"    {key}: {type(val).__name__} = {val_str}")
    
    if num_samples > 1 and len(data) > 1:
        print(f"\n  前 {min(num_samples, len(data))} 個樣本的 SMILES:")
        for i in range(min(num_samples, len(data))):
            smiles = data[i].get('smiles', 'N/A')
            label = data[i].get('label', 'N/A')
            print(f"    [{i}] smiles={str(smiles)[:60]}... label={label}")
    
    print(f"=== 檢視完成 ===\n")


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='將 CSV 轉換為 evaluate_mpp.py 所需的 PKL 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 轉換 bace CSV → PKL
  python csv_to_pkl.py --taskname bace --dataroot ./data/mpp/raw/ --datatarget ./data/mpp/pkl_test/
  
  # 僅驗證
  python csv_to_pkl.py --verify --new_pkl ./data/mpp/pkl_test/bace.pkl --ref_pkl ./data/mpp/pkl/bace.pkl
  
  # 轉換 + 驗證
  python csv_to_pkl.py --taskname bace --dataroot ./data/mpp/raw/ --datatarget ./data/mpp/pkl_test/ \\
      --verify --ref_pkl ./data/mpp/pkl/bace.pkl
  
  # 檢視 PKL 結構
  python csv_to_pkl.py --inspect ./data/mpp/pkl/bace.pkl
        """
    )
    
    parser.add_argument('--taskname', type=str, default=None,
                        help='任務名稱 (bace, bbbp, clintox, sider, tox21, toxcast, freesolv, esol, lipophilicity)')
    parser.add_argument('--dataroot', type=str, default='./data/mpp/raw/')
    parser.add_argument('--datatarget', type=str, default='./data/mpp/pkl_test/')
    parser.add_argument('--num_cores', type=int, default=4)
    parser.add_argument('--smiles_timeout_sec', type=int, default=60,
                        help='單筆 SMILES 最長處理秒數，超時自動跳過')
    parser.add_argument('--failed_csv_path', type=str, default=None,
                        help='失敗樣本清單輸出路徑 (CSV)')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--new_pkl', type=str, default=None)
    parser.add_argument('--ref_pkl', type=str, default=None)
    parser.add_argument('--inspect', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_pkl(args.inspect)
        return
    
    output_path = None
    
    if args.taskname:
        output_path = process_csv_to_pkl(
            task_name=args.taskname,
            csv_dir=args.dataroot,
            target_pkl_dir=args.datatarget,
            num_cores=args.num_cores,
            smiles_timeout_sec=args.smiles_timeout_sec,
            failed_csv_path=args.failed_csv_path
        )
    
    if args.verify:
        new_pkl = args.new_pkl or output_path
        ref_pkl = args.ref_pkl
        
        if new_pkl is None:
            print("[ERROR] 驗證模式需指定 --new_pkl 或先執行轉換 (--taskname)")
            sys.exit(1)
        if ref_pkl is None:
            if args.taskname:
                ref_pkl = os.path.join('./data/mpp/pkl/', f'{args.taskname}.pkl')
                print(f"[INFO] 自動使用參考 PKL: {ref_pkl}")
            else:
                print("[ERROR] 驗證模式需指定 --ref_pkl")
                sys.exit(1)
        
        if not os.path.exists(new_pkl):
            print(f"[ERROR] 新 PKL 不存在: {new_pkl}")
            sys.exit(1)
        if not os.path.exists(ref_pkl):
            print(f"[ERROR] 參考 PKL 不存在: {ref_pkl}")
            sys.exit(1)
        
        inspect_pkl(new_pkl, num_samples=2)
        inspect_pkl(ref_pkl, num_samples=2)
        
        is_match = verify_pkl(new_pkl, ref_pkl)
        sys.exit(0 if is_match else 1)
    
    if not args.taskname and not args.verify and not args.inspect:
        parser.print_help()


if __name__ == '__main__':
    main()
