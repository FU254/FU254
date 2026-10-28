# 单教师四组最佳模型：补测冻结测试集

此前的 `single_teacher_overfit` 研究只评估训练集和验证集，日志中的 `validation_only` 不是测试成绩。此入口读取已经保存的 A/B/C/D 最佳模型，使用原测试患者、保存的预处理和验证集阈值，补算患者级测试指标；不训练、不加载教师、不重新选择轮次或阈值。

四组依次为 A纯监督、B完整蒸馏、C仅软标签蒸馏、D完整蒸馏加L2-SP。默认统一评估四组，不根据验证或测试成绩丢弃较差的对照。源研究文件、模型和训练曲线保持原样，测试结果另存。

## 使用

解压 `FU254-single-teacher-test-evaluation.zip` 后，将整个 `single_teacher_test_evaluation_update` 文件夹上传到 `/home/cfu/EVA-X-main/`。无需运行安装器，也无需覆盖旧训练代码。先查看入口是否存在：

```bash
ls -l /home/cfu/EVA-X-main/single_teacher_test_evaluation_update/run_single_teacher_test_evaluation.py
```

在服务器整段执行：

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
mkdir -p runs/single_teacher_overfit
single_test_log="runs/single_teacher_overfit/test_$(date +%Y%m%d_%H%M%S).log"
nohup /opt/anaconda3/envs/evax_lora/bin/python -u \
    single_teacher_test_evaluation_update/run_single_teacher_test_evaluation.py \
    --project /home/cfu/EVA-X-main \
    --study /home/cfu/EVA-X-main/runs/single_teacher_overfit/studies/study_20261028T074642Z_e0e32fba \
    --device auto \
    > "$single_test_log" 2>&1 < /dev/null &
echo "评估PID：$!；日志：$single_test_log"
tail -n 40 -f "$single_test_log"
```

`auto` 在符合条件的空闲GPU上逐组推理；不符合条件时使用CPU，不占用繁忙卡。也可用 `--device cuda:具体编号` 明确指定确认可用的逻辑GPU编号。这里是推理和置信区间计算，不包含训练轮次。`Ctrl+C` 只退出日志查看，后台评估继续。

程序先核对四组检查点SHA256、原研究协议、患者划分、学生初始化和保存的阈值；四组测试患者必须相同，且不能与训练／验证患者重叠。影像输入通过原 `inference_dataset` 重建共享视野，保留教师尺寸的第一次缩放／裁剪和学生尺寸的第二次缩放。

完成后控制台明确打印 `TEST patient results`，包括 ROC-AUC、PR-AUC、AP、Sensitivity、Specificity 及患者bootstrap置信区间。本次实际研究测试集为399名患者；程序按保存的原始划分读取，不创建新划分。

## 输出与再次查看

输出目录为：

```text
/home/cfu/EVA-X-main/runs/single_teacher_overfit/studies/study_20261028T074642Z_e0e32fba/report/test_evaluation_seed42/
```

- `comparison.csv`：四组测试指标、置信区间和固定阈值。
- `A/`、`B/`、`C/`、`D/`：各组测试预测、患者预测、`metrics.json`、ROC／PR／校准曲线与评估凭据。

之后执行下列命令只读取已核验结果，不重新推理：

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
/opt/anaconda3/envs/evax_lora/bin/python \
    single_teacher_test_evaluation_update/run_single_teacher_test_evaluation.py \
    --project /home/cfu/EVA-X-main \
    --study /home/cfu/EVA-X-main/runs/single_teacher_overfit/studies/study_20261028T074642Z_e0e32fba \
    --show-metrics
```

重跑补测命令也会复用已完成且指纹一致的组。不同进程不能同时评估同一研究。测试产物不写入训练run的 `test/`，因此原诊断研究的验证报告和恢复校验仍可使用。

测试指标无需与验证指标相同。保留所有组的结果，不以测试成绩重新挑选轮次、阈值、超参数或种子。单次seed42结果可报告为本次运行成绩，不能代表跨种子的稳定性。
