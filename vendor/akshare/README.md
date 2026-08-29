# AKShare单provider派生wheel

本目录记录M6认证使用的AKShare派生wheel来源和唯一补丁。上游固定为
`akfamily/akshare`的annotated tag`release-v1.18.88`，commit为
`02f358a520e4fb1cae72ed70bb2e18ba61738800`。

上游1.18.88在Linux同时声明`py-mini-racer`和`akracer`；两个发行包会写入不同内容的
`py_mini_racer/__init__.py`和`py_mini_racer/py_mini_racer.py`，结果依赖安装顺序。
补丁不修改任何AKShare运行源码，只做两件事：

1. 将本地版本标识改为`1.18.88.post1`；
2. 将平台分裂且重叠的三个依赖统一为跨平台`mini-racer==0.14.1`。

构建时必须设置为上游commit时间戳的`SOURCE_DATE_EPOCH=1786630444`，从上述commit的干净checkout应用
`akshare-single-provider.patch`，再使用Python3.10执行：

```text
python -m pip wheel --no-deps --no-build-isolation --wheel-dir <wheel-output> <patched-checkout>
```

构建两次的wheel必须字节一致。生成物
`vendor/wheels/akshare-1.18.88.post1-py3-none-any.whl`的SHA-256为
`5513241579a1b830c1d0f0067162ce78991c02060d2a9aaab8fd65fd671d629a`。
