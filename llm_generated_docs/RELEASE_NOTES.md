# v0.1.1

- 非零 NPU IOMMU domain 的驱动版本门槛调整为 RKNPU 0.9.0，允许启用 IOMMU 的 0.9.6 驱动初始化模型。
- IOMMU domain 初始化失败时显示 domain 编号、最低驱动版本和 IOMMU 要求。

ARM64 二进制适用于 Debian 11 / glibc 2.31 或更新的系统，依赖 libstdc++6、libgomp1 和 RK3588 NPU DRM 驱动；模型独立提供。附件包含二进制和固定依赖的源码包。
