# Jing gsync 运行指南

这个仓库本身就是被 `gsync` 监控的目标仓库。Jing 需要在 Minerva 上另外 clone 一份
`gsync` 工具，然后用那个工具来监控本仓库。

## 1. Jing 先准备 gsync 工具

```bash
git clone git@github.com:thynics/gsync.git /path/to/gsync
cd /path/to/gsync
chmod +x gsync.sh gsync_worker.sh
```

## 2. Jing clone 本项目作为 broker 工作目录

建议使用一个专门给 `gsync` 用的 clone，不要和日常开发目录混用：

```bash
git clone <this-repo-url> /path/to/broker-clone
cd /path/to/broker-clone

git config user.name "gsync-broker"
git config user.email "gsync-broker@local"
```

这里的 `/path/to/broker-clone` 就是本项目的 clone，例如：

```bash
/cephyr/users/chjing/Sweep_LLM_Jobs_broker
```

## 3. 启动 broker 和 worker

broker 用普通权限启动：

```bash
cd /path/to/gsync
./gsync.sh start /path/to/broker-clone
./gsync.sh status /path/to/broker-clone
```

worker 用高级权限启动：

```bash
cd /path/to/gsync
sudo ./gsync_worker.sh start /path/to/broker-clone
sudo ./gsync_worker.sh status /path/to/broker-clone
```

也就是说，Jing 需要高级权限运行的是：

```bash
sudo ./gsync_worker.sh start /path/to/broker-clone
```

不要用 `sudo` 启动 broker。

## 4. 提交测试任务

本仓库已经包含测试任务脚本：

```text
jobs/dvfs-smoke-test/run.sbatch
```

第一次确认 broker 和 worker 已经启动后，再创建 `READY` 标记并 push：

```bash
cd /path/to/broker-clone
touch jobs/dvfs-smoke-test/READY
git add jobs/dvfs-smoke-test
git commit -m "jobs: submit dvfs smoke test"
git push
```

注意：`READY` 必须最后创建。如果在 broker 第一次启动前就已经存在 `READY`，
`gsync` 会把它当成历史任务，不会自动提交。

## 5. 查看结果

```bash
cd /path/to/broker-clone
git pull
cat jobs/dvfs-smoke-test/status.json
cat jobs/dvfs-smoke-test/status.log
```

也可以查看 Slurm 输出：

```bash
ls jobs/dvfs-smoke-test/slurm-*.out jobs/dvfs-smoke-test/slurm-*.err
```

broker/worker 日志在：

```text
/path/to/broker-clone/.broker/broker.log
/path/to/broker-clone/.broker/slurm_worker.log
```

## 6. 停止服务

```bash
cd /path/to/gsync
./gsync.sh stop /path/to/broker-clone
sudo ./gsync_worker.sh stop /path/to/broker-clone
```
