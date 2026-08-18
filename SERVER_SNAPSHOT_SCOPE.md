# 服务器快照范围

本文件说明 GitHub 仓库 `tandem_hrl_current_progress` 的服务器快照边界。

## 已包含

- `/root/gpufree-data/tandem_hrl_620d798` 项目目录的完整当前内容。
- `/root/gpufree-data/legacy_zyb_check` 稳定性脚本、历史日志、CSV、reference checkpoint 和惯量 USD。
- `/root/gpufree-data/zyb_v0_model_1023.pt`、`zyb_v0_lower_init_zero_wheel.pt`。
- `/root/gpufree-data/zyb_real_grasp_8417524` 和真实抓取相关校准结果。
- `/root/gpufree-data` 下的历史训练/评估目录和小型输出文件。
- `tandem_620d798_gitarchive.tar` 的分片。单个分片小于 GitHub 普通文件限制，恢复方法见下文。

完整 `server_snapshot/` 目录作为 GitHub Release 资产
`server_snapshot_20260818.tar` 上传；主分支保留可直接 clone 的代码和核心参考资产。

## 不包含

- `/workspace/isaaclab`、Isaac Sim 安装、系统缓存和用户 home 下的缓存；这些不是项目源代码，复现者需要自行准备 Isaac Sim 5.1 / Isaac Lab 2.3.2。
- `lost+found` 文件系统保留目录。
- 服务器登录密码、GitHub token、桌面的 `1.txt`。

## 原始归档恢复

进入 `reference_server_data/server_snapshot/tandem_620d798_gitarchive/` 后执行：

```bash
cat part-* > tandem_620d798_gitarchive.tar
tar -tf tandem_620d798_gitarchive.tar | head
```

该归档是历史服务器归档，项目当前可复现代码以仓库根目录的源代码为准。
