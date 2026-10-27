# 接收已经完成的 P-R 结果

本工具用于独立 P-R 已完成、原队列因接收进程竞态而把 P-R 标记失败的情况。它兼容先前已经安装的并行模块，不替换训练代码，也不调用训练、GPU 预检查或测试评估。

已确认的竞态是：独立进程完成保存后释放组锁、删除进程标记，原队列随即获取锁；此时独立进程可能仍在打印最后几行，因此被旧的进程检查误判为另一个未受保护的训练任务。代码库已经修正为优先严格核验并接收完整结果，再在准备新训练时检查其他进程。

恢复本批结果时无需更新并行模块。工具同时获取原研究锁和本组锁，要求独立回执状态为 `completed`，核验配置、教师来源、患者划分、学生初始化、检查点 SHA256、完整蒸馏状态及已保存测试文件。可用额外参数固定期望的检查点 SHA256。任何核验失败均不会启动训练。

## 上传并执行

解压 `FU254-teacher-weight-result-recovery.zip`，将整个 `teacher_weight_result_recovery` 文件夹上传到 `/home/cfu/EVA-X-main/`。然后执行：

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
/opt/anaconda3/envs/evax_lora/bin/python \
    /home/cfu/EVA-X-main/teacher_weight_result_recovery/recover_teacher_weight_result.py \
    --project /home/cfu/EVA-X-main \
    --study /home/cfu/EVA-X-main/runs/teacher_weight_study/studies/study_20261026T134529Z_e3d7615f \
    --group P-R \
    --expected-checkpoint-sha256 4d52190cb177c56d274f018c2d8cbdd67b59b075b8b27670c50390c984eed1c9
```

研究或本组仍被其他进程持锁时，工具会停止并保留现有任务。它不发送停止信号。成功恢复会保留原失败尝试，增加恢复审计记录，将 P-R 标记为 `resumed`，并重新生成六组比较报告；原始主记录备份到研究目录下的 `study.json.before_result_recovery_*.bak`。重复执行会确认已经登记的同一结果，不重复改写记录。

恢复成功后生成六组曲线：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_teacher_weight_experiments.py \
    --project /home/cfu/EVA-X-main \
    --plot-curves \
    --study /home/cfu/EVA-X-main/runs/teacher_weight_study/studies/study_20261026T134529Z_e3d7615f
```

这里读取现有预测、指标和历史，不重新计算测试集，也不选择另一轮训练或另一个随机种子。最佳与最终轮次仍分别报告；同口径的训练与验证 Focal loss、患者 ROC-AUC 曲线用于判断拟合趋势。
