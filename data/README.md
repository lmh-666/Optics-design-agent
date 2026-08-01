# Data Directory

完整镜头数据集、CODE V `.dat` 文件、`.seq` 文件和 ray tracing cache 不随公开仓库提交。

公开仓库只保留目录结构：

- `data/dat/`
- `data/codev_blocks/`
- `data/eval/`
- `data/raytrace_cache/`

本地运行完整系统时，请将私有数据放回对应目录，并通过环境变量指定主数据文件：

```bash
set LENS_DATA_PATH=.\data\lens_data.csv
```

主镜头表至少需要包含以下字段或等价别名：

- `lens_id`
- `f_number`
- `half_fov`
- `total_length`
- `focal_length`

