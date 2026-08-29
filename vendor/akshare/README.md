# AKShare单provider派生wheel

M6认证固定使用PyPI官方文件：

- URL：`https://files.pythonhosted.org/packages/8e/24/f4a3d9d58993a67bf3ead06e44ad9dcd062eaa1e2b719be4be8e7e2646cf/akshare-1.18.88-py3-none-any.whl`
- SHA-256：`ba0b06ea2d341122e2ef8ed5e4982ff5925f01111e63d7940c8a01aa684578c0`
- 源码来源：`akfamily/akshare`的轻量tag`release-v1.18.88`
- tag所指commit：`02f358a520e4fb1cae72ed70bb2e18ba61738800`

`release-v1.18.88`是轻量tag，不是annotated tag；源码commit仅用于来源追踪。派生wheel直接读取上面
固定哈希的官方wheel，不再从源码checkout构建，因此Git换行配置和setuptools版本不会改变payload。

官方1.18.88元数据同时声明三个会覆盖同一`py_mini_racer`命名空间的provider。本项目的
metadata-only派生规则是：

1. 将dist-info目录和`METADATA`中的distribution版本改为`1.18.88.post1`；
2. 将三个冲突的`Requires-Dist`替换为唯一`mini-racer==0.14.1`；
3. 重算`RECORD`；
4. 其余entry逐字节复制，包含`akshare/_version.py`、全部运行代码、资源、`WHEEL`和LICENSE。

因此`importlib.metadata.version("akshare")`为`1.18.88.post1`，而官方运行payload中的
`akshare.__version__`保持`1.18.88`。官方wheel和派生wheel都没有独立NOTICE文件；名称中含
`notice`的Python模块不属于NOTICE许可文件。LICENSE payload与官方逐字节一致。

[重打包工具](../../tools/repack_akshare_wheel.py)仅使用Python标准库，按固定entry顺序、时间戳、
Unix普通文件权限和`ZIP_STORED`写出，避免zlib和构建工具差异。仓库派生wheel的SHA-256固定为：

`de6d7f77008299b5c96e13559542e153ac58d0780523c0d5b45694ce97e2099f`

正式生成和认证命令：

```text
python tools/repack_akshare_wheel.py fetch --output <work>/akshare-1.18.88-py3-none-any.whl
python tools/repack_akshare_wheel.py build --source <work>/akshare-1.18.88-py3-none-any.whl --output vendor/wheels/akshare-1.18.88.post1-py3-none-any.whl
python tools/repack_akshare_wheel.py certify --work-dir <work>/certify --committed-wheel vendor/wheels/akshare-1.18.88.post1-py3-none-any.whl
```

`certify`会重新下载并校验官方哈希、独立重建两次、要求两次结果与仓库wheel三者哈希相同，
同时检查entry拓扑、RECORD、LICENSE和所有非`METADATA`payload。CI在Windows和Linux的
Python3.10、3.11、3.12矩阵全部执行该认证。
