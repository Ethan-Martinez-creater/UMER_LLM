# round_021_cross_view_consistency

方案标识：`round_021_cross_view_consistency`

Round 018 五折已经达到 Accuracy 0.8864、Weighted-F1 0.8869、Macro-F1 0.8798，但 Rumor-F1 只有 0.8516。Round 019/020 表明继续增加融合头复杂度会放大验证—测试落差。本轮回到文本表示泛化：每个训练 event 不再只随机取一个文本视图，而是无放回抽取两个不同视图，同时监督两个视图，并对二者预测加入广义 Jensen-Shannon 一致性约束。推理仍统一平均四个标准视图，模型仍是三分支共同训练并保存单一 checkpoint。

为保持资源和优化尺度可比，物理 event batch 从 8 调为 4、每步文本序列仍为 8，梯度累积从 8 调为 16，等效 event batch 仍为 64。其余设置完全沿用 Round 018：DeBERTa-v3-large clean 初始化、线性联合融合、平方根逆频率类别权重、严格训练折 leave-one-out 检索记忆。

近期文献审计显示，GCML（EMNLP 2025）针对跨主题泛化采用目标域少样本元适配，但直接使用本实验验证折标签训练会破坏严格协议；EIN（WWW 2025）依赖额外 LLM 回复立场伪标签；EPSE（Neurocomputing 2026）强调传播序列与事件级情绪，但当前代理情感特征已在 HAT-KI 筛选中仅获极小增益。因此本轮不声称复现这些方法，而是针对本项目已存在的训练/推理视图不一致做最小、可审计的直接修正。

固定 `train_view_count=2`、一致性权重 `0.5`，先运行 seed 2000 / fold 1；验证集选择 checkpoint 和阈值，测试集仅在早停后评估一次。只有相对 Round 018 同折有实质增益才扩展五折。

## fold 1 正式筛选结果

seed 2000 / fold 1 在 epoch 15 达到验证最优 0.9125/0.9128/0.9071/0.8849，epoch 22 正常早停；恢复单一最佳 checkpoint 后，测试为 Accuracy 0.9035、Weighted-F1 0.9038、Macro-F1 0.8977、Rumor-F1 0.8732。相对 Round 018 同折的 0.8911/0.8913/0.8842/0.8560，四项分别提升约 0.0125、0.0125、0.0135、0.0172。虽然 Rumor-F1 单折仍未达到 0.88，但增益显著且四项同步改善，因此扩展 seed 2000 严格五折。

seed 2000 五折四项均值全部严格超过 0.88 后，才允许使用 `run_additional_seed_fivefold.sh 3090` 和 `run_additional_seed_fivefold.sh 5090` 依次扩展其余两个 seed。脚本固定本轮全部正式参数、逐折断点续跑并在五折完成后生成各自的 `summary.json`；不得因中间结果修改配置。

## seed 2000 五折过程记录

fold 2 在 epoch 6 取得验证最优 0.9086/0.9095/0.9044/0.8845，随后按 patience 7 正常早停；恢复该 checkpoint 后测试为 Accuracy 0.8809、Weighted-F1 0.8819、Macro-F1 0.8752、Rumor-F1 0.8484。前两折 Rumor-F1 均值约 0.8608，说明验证—测试域偏移尚未解决。为保持正式五折协议，fold 3--5 仍按冻结配置继续，不根据中间测试结果调参。

## seed 2000 五折最终结果

五折测试均值/标准差为 Accuracy 0.8951±0.0101、Weighted-F1 0.8955±0.0099、Macro-F1 0.8890±0.0102、Rumor-F1 0.8629±0.0115。前三项严格超过 0.88，Rumor-F1 未达标，因此本轮不扩展 seed3090/5090。各折 Rumor-F1 为 0.8732/0.8484/0.8705/0.8493/0.8732；高低折交替出现，且所有折使用同一 runner、超参数、partition seed 与 training seed，不存在 fold1 专用训练逻辑。完整证据：`results/round_021_cross_view_consistency/fivefold_seed2000/summary.json`。

## fold2/fold4 低值诊断

外层 `StratifiedKFold` 只按标签分层，各测试折约含 480--481 个 rumor 和 804--805 个 non-rumor，因此 fold2/fold4 不是由少数类数量更少造成。相同划分的 Round018 最低折是 fold3，且五折 Accuracy 标准差仅 0.0034；Round021 相对 Round018 在 fold1/3/5 明显提升，fold2 小幅下降、fold4基本不变，说明低值并非固定“坏折”，而是跨视图一致性收益与训练随机性在不同划分上不均匀。

验证—测试落差也呈反向关系：fold2/fold4 最佳验证 Accuracy 为 0.9086/0.9125，但测试仅 0.8809/0.8848；fold3/fold5 验证仅 0.8852/0.8813，测试却达到 0.9012/0.9051。内层验证仅从 train-val 部分按标签抽取 10%，没有按新闻主题、事件簇、传播规模或回复数分层，因此 checkpoint 与阈值在外层测试折上存在分布错配。

五折固定 checkpoint 的仅诊断测试 oracle 阈值审计表明，fold2 的阈值从验证选择 0.61 调到测试 oracle 0.80 后 Rumor-F1 由 0.8484 升到 0.8601；fold4 从 0.44 调到 0.871 后由 0.8493 升到 0.8586，仍明显低于高折。五折 oracle 均值为 0.9002/0.9005/0.8942/0.8688，仍不能使 Rumor-F1 超过 0.88。故阈值校准偏移解释一部分，主要问题仍是外层测试上的类别可分性与泛化差异；oracle 仅用于诊断，不得用于论文正式选模或报告。服务器证据：`analysis/joint_threshold_reachability/round_021_seed2000_fivefold/summary.json`。

## partition/training seed 敏感性矩阵（2026-07-19 启动）

为分别估计训练随机性和五折划分随机性，冻结 Round21 的全部模型与训练参数，预声明四个新增组合：固定 `partition_seed=3090` 时运行 `training_seed=42/5090`；固定 `training_seed=2000` 时运行 `partition_seed=42/5090`。原结果 `partition_seed=3090, training_seed=2000` 作为参考，共形成五个组合。每个新增组合完整运行五折，不根据中间结果停止或调参。

统一入口 `launch_seed_sensitivity_matrix.sh` 只需启动一次；后台总驱动 `run_seed_sensitivity_matrix.sh` 按 `p3090/t42 → p3090/t5090 → p42/t2000 → p5090/t2000` 严格串行，共20次训练。单组合脚本支持按 `result.json` 断点续跑。输出根目录为 `results/round_021_cross_view_consistency/seed_sensitivity/`，全部完成后自动生成 `matrix_summary.json`，同时分别报告固定 partition 时 training seed 的五折均值极差、固定 training 时 partition seed 的五折均值极差。首次启动 PID 为 `452428`；结果完成前不做选择性分析。

## Ma-Weibo 迁移实验（partition 3090 / training 2000）

在 Ma-Weibo 上保持 Round21 正式配置：DeBERTa-v3-large clean 初始化、三分支线性联合融合、双训练视图与权重 0.5 的跨视图一致性、平方根逆频率类别权重、严格训练折检索记忆、四视图验证/测试、`max_epochs=30`、`patience=7`、五折 `partition_seed=3090`、`training_seed=2000`。原始数据与预处理图只读使用；新增 manifest、日志、checkpoint 与结果均写入部署工作区（部署位置见内部指南，不随本仓库分发）。

默认物理 event batch 为 4、每个 event 两个文本视图、梯度累积 16，等效 event batch 为 64。若日志明确出现 CUDA OOM，五折驱动仅将物理 event batch 降至 2 并将梯度累积增至 32，仍保持等效 batch 64；OOM 日志保留用于审计。统一入口为 `launch_maweibo_fivefold.sh`，各折按 `result.json` 支持断点续跑，结果目录为 `results/round_021_cross_view_consistency/maweibo_partition3090_train2000/`。

启动前 manifest 核验覆盖 4664 个事件，标签分布为 non-rumor 2351、rumor 2313。五折后台驱动于 2026-07-19 启动，PID 为 `471533`；首折训练子进程 PID 为 `471542`。

首次启动因 Ma-Weibo `graph_final` 未保存 `node_ids` 而在文本视图加载阶段退出，并非 CUDA OOM。修复方式是在 Ma-Weibo 专用路径按预处理协议重建节点顺序：过滤至 240 小时、按时间升序排列、再按图中 `num_nodes` 截断；PHEME 路径保持原逻辑。失败日志保留为 `fold_1.node_ids_failure.log` 和 `launcher.node_ids_failure.log`。用户暂停后，于 2026-07-19 重新启动五折，主 PID `511115`、fold1 PID `511128`；4664 个事件映射全部通过，epoch1 已完成，物理 event batch 4 未发生 OOM。
