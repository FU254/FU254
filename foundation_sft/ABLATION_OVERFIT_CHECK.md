# 六组消融：不重新训练的过拟合检查

读取已经完成的六组消融批次，对每组 `best.pt` 和 `last.pt` 分别在无随机增强的训练集、验证集上重新推理。分类损失始终使用该组保存的同一 Focal 配置，不计入 KD 或正则项。本检查不加载教师、不更新模型、不重新选阈值、不在测试集上推理。

本轮对应：A纯监督、B软标签、C软标签+全局、D软标签+局部、E全局+局部、F完整蒸馏。默认检查本批六组seed42。

## 上传与启动

解压 `FU254-ablation-overfit-check.zip`，将完整的 `distillation_overfit_check_update` 文件夹上传到 `/home/cfu/EVA-X-main/`。包中两个Python文件均需上传：检查入口，以及复用的设备选择／文件校验辅助文件。原训练代码无需覆盖。

```bash
ls -l /home/cfu/EVA-X-main/distillation_overfit_check_update/check_distillation_overfitting.py
```

然后执行：

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
mkdir -p runs/distillation_ablation
ablation_check_log="runs/distillation_ablation/overfit_check_$(date +%Y%m%d_%H%M%S).log"
: > "$ablation_check_log"
nohup /opt/anaconda3/envs/evax_lora/bin/python -u \
    distillation_overfit_check_update/check_distillation_overfitting.py \
    --project /home/cfu/EVA-X-main \
    --suite /home/cfu/EVA-X-main/runs/distillation_ablation/suites/suite_20261028T121626Z_2cb70ef6 \
    --device auto \
    > "$ablation_check_log" 2>&1 < /dev/null &
echo "检查PID：$!；日志：$ablation_check_log"
tail -n 40 -F "$ablation_check_log"
```

`auto` 选择符合空闲显存和利用率条件的GPU，否则使用CPU；也可指定已确认空闲的逻辑设备，例如 `--device cuda:0`。这里仅有推理，不会出现新的训练轮次。`Ctrl+C` 只退出日志查看，后台检查继续。

## 看哪些指标

每组输出最佳、最后检查点的：

- `train_loss`、`val_loss`：同一 Focal 定义，均为无随机增强的评估损失，按影像平均。
- `train_roc_auc`、`val_roc_auc`：患者级ROC-AUC。
- `train_pr_auc`、`val_pr_auc`：患者级梯形PR面积，与AP不同。
- `auc_gap = train_roc_auc - val_roc_auc`。
- `loss_gap = val_loss - train_loss`。

结合完整验证历史判断，而不设置统一的gap阈值：

1. 干净训练表现继续改善，验证损失在一段时间内升高、验证AUC或PR下降，支持后期过拟合。
2. 训练集已高度拟合、验证长期停滞且差距持续，提示泛化受限和过拟合风险；单个差值仍不能确定原因。
3. 训练和验证仍同步改善，说明当前仍有验证收益；不能只因训练AUC接近1就断言严重过拟合。
4. 差距缩小但验证表现更差，不能据此认定问题解决，也可能是欠拟合或约束过强。

最佳检查点按最高验证ROC选择，因此最佳到最后的ROC下降带有选择效应，不能单独作为过拟合的证明。应同时看训练变化、验证损失、逐轮趋势和不同指标。验证集仅24名阳性患者，短期波动需要谨慎解释。

## 曲线的限制

本次只保存最佳和最后模型，无法还原中间每一轮的干净训练指标。新图只绘制这两个检查点的训练散点，不连接成虚构的逐轮曲线；验证曲线来自真实保存的 `history.json`，包含所有轮次。原训练总损失含蒸馏项，而且使用训练增强，不能直接与验证Focal相减。

程序打印准确输出目录，在原批次 `report/` 下新建独立的 `overfit_check_*` 子目录，保存检查表、来源指纹和Focal／ROC／PR图。原模型、原训练历史和已有测试成绩保留。再次执行会新建检查目录，只重复只读推理，不重复训练。
