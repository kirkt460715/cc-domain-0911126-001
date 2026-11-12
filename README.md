# 学龄人口与学位压力

本项目使用 Python 3.11 及以上版本，仅依赖标准库，无需安装第三方包。数据文件均为本地合成记录，不含真实个人或机构资料。

cohorts 校验叶子地区人口和迁移守恒；forecast 管理情景与历史点预测回测；school_pressure 计算学段容量及行政层级汇总；releases 保存运行、发布与校验摘要。

人口只登记在叶子地区。情景范围是确定性假设范围。已有回测按发布时间和目标年配对，不将人为情景范围解释为预测置信区间。

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
