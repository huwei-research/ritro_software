# RITRO 软件交付

本目录保存由 `Research/ritro` 导出的确定版本软件。持续开发、可编辑论文、实验与研究状态均在研究主仓维护，唯一研究状态为 `Research/ritro/STATE.md`，统一日志为 `Research/ritro/research_log.md`。当前本地交付情况见[STATE.md](STATE.md)。

交付文件由研究主仓的 `scripts/export_project_delivery.py` 生成，清单见 `delivery_manifest.json`。再次导出前检查独有修改；禁止直接覆盖被独立编辑的交付文件。普通使用不依赖私有研究仓。原始研究数据、审阅记录和历史私有材料不复制进开源软件目录。

本目录已准备不代表已经提交期刊、上传arXiv或发布GitHub；不自动commit、push或修改远端。第三方文件保留各自许可。

<!-- project-ownership -->
```json
{"maintenance": "delivery", "project_id": "ritro", "source_root": "Research/ritro", "status": "aligned"}
```
<!-- /project-ownership -->

<!-- unified-project-layout -->
## 统一目录职责

遵循[三类目录统一规范](../../research_harness/policies/research_delivery_layout.md)；本仓实际职责和路径映射见[project_layout.json](project_layout.json)。研究状态与交付状态分别维护。导出必须比对独有修改、记录来源散列并独立验证；迁移完成前保持真实待归并状态。独立克隆缺少工作区规范时，继续遵循本文件已有的主源、资产保护及验证要求。
<!-- /unified-project-layout -->

<!-- manuscript-review-delivery -->
## 文件组织与论文审阅交付

- 文件归属和目标命名遵循[详细目录](../../research_harness/policies/project_file_catalog.md)。算法、实验、输入、原始运行、分析和精简数据各归其指定位置；现有路径迁移前保持真实映射，保留历史资产。
- 论文按版本独立保存完整依赖，正文以章节组织；每轮论文交付从同版主源自动生成PDF、完整单文件review.tex、review.zip和source.zip，写入该版output，索引记录实际位置。
- 单文件包含正文、附录、宏定义、图注、表格和实际文献条目；保留TeX语义，图件及样式随审阅包提供。生成稿不独立编辑，外部意见回写主源后重新生成。
- 交付前核对内容与引用，并在隔离目录编译审阅包；记录源版本、格式、输入输出摘要、命令和实际验证结果。未生成或未验证的产物明确列为待完成。日常编辑不逐次打包，规则维护不触发论文改版。
- 完整要求见[审阅交付规则](../../research_harness/policies/manuscript_review_delivery.md)。Publish只接收确定副本，Public保持软件职责；独立克隆缺少工作区规则时仍执行本节摘要。交付回复链接PDF和review.tex，按需链接两个压缩包。
<!-- /manuscript-review-delivery -->
