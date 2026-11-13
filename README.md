# 学龄人口与学位压力

本项目使用 Python 3.11 及以上版本，仅依赖标准库，无需安装第三方包。数据文件均为本地合成记录，不含真实个人或机构资料。

cohorts 校验叶子地区人口和迁移守恒；forecast 管理情景与历史点预测回测；school_pressure 计算学段容量及行政层级汇总；releases 保存运行、发布与校验摘要；studies 管理可保存、可复现的联合预测研究。

人口只登记在叶子地区。情景范围是确定性假设范围。已有回测按发布时间和目标年配对，不将人为情景范围解释为预测置信区间。

## 联合预测研究

studies 把历史点预测误差按预测跨度组织成覆盖全部叶子单元的联合残差向量，以发布年份为整体重采样，保留同一冲击在各地区间的关联，避免把同一年的迁移冲击当成彼此独立后逐乡镇相加。

* 实际人数带 `first_known_year`；每个预测起点只使用当时已经可知且目标已到期的误差，未来才可知的修订不进入旧研究。
* 缺地区的历史向量逐条列出原因（缺失预测、缺失实际、未到期、尚未可知），绝不补零后混入。
* 训练、区间校准、最终回测的发布年份彼此分开；校准用有限样本 split-conformal 分位差，回测在从未参与拟合的年份上报告覆盖率、平均宽度和区间分数。
* 每次模拟先形成非负整数叶子需求（负需求截断为零并计数），再由同一次模拟聚合到祖先地区并调用既有 `assess` 容量口径；分位数不要求可加，但每级人数逐次模拟核对相符。
* 输出叶子与全县的中位数、指定覆盖率区间、容量超越频率与条件超载分布；`as_observed`、`aligned`（同向）、`opposed`（反向）三种依赖形态演示全县风险如何随地区误差关联变化。
* 历史不足的跨度可选择停止概率输出，或按说明把共享同一叶子单元空间的跨度合并重采样。
* 研究归档抽样次数、种子、训练年份、排除理由、输入摘要与每次重采样索引；同一研究重开复现完全相同的样本，补到新实际值（输入摘要变化）时必须另建研究。

```sh
python3 cohort_cli.py study-run study_input.json --output study.json
python3 cohort_cli.py study-show study.json            # 精简视图
python3 cohort_cli.py study-show study.json --full     # 含逐次重采样索引的完整归档
python3 cohort_cli.py study-verify study.json --input study_input.json
python3 cohort_cli.py study-demo --fixture fixtures/planning_history.json
```

`study-demo` 在临时目录内运行主研究与四个边界情景：未来才可知的修订、容量为零、历史不足（停止与合并两种策略），结束后自动清理。

在项目根目录运行完整测试：

```sh
python3 -m unittest discover -s tests -v
```

编译检查：

```sh
python3 -m compileall -q .
```

实际演示：

```sh
python3 cohort_cli.py demo --fixture fixtures/county.json
```

演示在临时目录内建立本地状态，结束后自动清理。测试、编译与演示在依赖已准备的环境内均不下载数据或软件。输入错误以非零退出码及错误对象报告。更多子命令可通过对应入口的 --help 查看。

容器环境可执行 `docker build -t local-analysis .`，再运行 `docker run --rm --network none local-analysis`。镜像固定基础内容摘要，构建后运行只访问本地文件。完整测试与演示也可直接使用已有 Python 环境。
