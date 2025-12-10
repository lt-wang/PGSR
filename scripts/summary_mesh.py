import os
import glob
import json
import pandas as pd
import re


def find_all_results_json(root_dir):
    # 递归查找所有 results.json 文件
    return glob.glob(os.path.join(root_dir, '**', 'results.json'), recursive=True)


def extract_scan_number(scanid):
    match = re.match(r'scan(\\d+)', scanid)
    return int(match.group(1)) if match else float('inf')


def main(root_dir):
    results_files = find_all_results_json(root_dir)
    data = []
    for result_file in results_files:
        with open(result_file, 'r') as f:
            result = json.load(f)
        # 获取 scan_id（路径的第一个部分）
        rel_path = os.path.relpath(os.path.dirname(result_file), root_dir)
        scan_id = rel_path.split(os.sep)[0]
        mean_d2s = result.get('mean_d2s', 'N/A')
        mean_s2d = result.get('mean_s2d', 'N/A')
        overall = result.get('overall', 'N/A')
        data.append({
            'ScanID': scan_id,
            'mean_d2s': mean_d2s,
            'mean_s2d': mean_s2d,
            'overall': overall
        })
    df = pd.DataFrame(data)

    # 计算均值，排除 'N/A'
    df_numeric = df.copy()
    for col in ['mean_d2s', 'mean_s2d', 'overall']:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    avg_d2s = df_numeric['mean_d2s'].mean()
    avg_s2d = df_numeric['mean_s2d'].mean()
    avg_overall = df_numeric['overall'].mean()
    avg_row = pd.DataFrame([{'ScanID': 'Average', 'mean_d2s': avg_d2s, 'mean_s2d': avg_s2d, 'overall': avg_overall}])

    # 只对 scan 开头的行按数字排序
    scan_df = df[df['ScanID'].str.startswith('scan')].copy()
    scan_df['scan_num'] = scan_df['ScanID'].apply(lambda x: int(x[4:]) if x[4:].isdigit() else float('inf'))
    scan_df = scan_df.sort_values('scan_num')
    scan_df = scan_df.drop(columns=['scan_num'])
    # 其他行（如非 scan 开头）
    other_df = df[~df['ScanID'].str.startswith('scan')]
    # 合并，最后加上均值
    df_with_avg = pd.concat([scan_df, other_df, avg_row], ignore_index=True)
    print(df_with_avg.to_string(index=False))
    # 保存为 CSV
    out_csv = os.path.join(root_dir, 'results_summary.csv')
    df_with_avg.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="汇总 release 目录下所有 results.json 的结果")
    #parser.add_argument('--release_dir', '-r', default='exp_dtu/release', help='release 目录路径')
    parser.add_argument("--model_path", "-m", help="model path",default="output/dtu")
    args = parser.parse_args()
    main(args.model_path)
