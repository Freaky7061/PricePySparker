# PricePySparker
基于监督学习的租房价格分析与预测系统

# 🏠 基于 PySpark 的租房价格分析与预测系统

**一个使用监督学习方法的租房价格数据分析与预测项目**

![PySpark](https://img.shields.io/badge/PySpark-3.3.0-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.2-orange)

## 📌 项目概述
本项目利用 **PySpark** 和机器学习技术，对租房市场价格数据进行深入分析，并构建预测模型，为以下场景提供数据支持：
- **房东**：制定合理租金策略
- **租客**：评估租房价格合理性
- **政策制定者**：了解住房市场趋势

## 🚀 主要功能
- **数据预处理**：处理缺失值、异常值和特征缩放
- **探索性数据分析(EDA)**：可视化价格分布、特征相关性等
- **机器学习模型**：
  - 线性回归
  - 决策树
  - 随机森林
- **模型评估**：比较RMSE、MAE和R²评分
- **价格优化**：基于特征推荐最优租金价格

## 📥 安装指南

### 1. 克隆仓库
```bash
git clone https://github.com/Freaky7061/PricePySparker.git
cd PricePySparker
```

### 2. 创建Python环境(推荐Anaconda)
```bash
conda create -n rentpred python=3.8
conda activate rentpred
pip install -r requirements.txt
```

### 3. 安装PySpark
```bash
pip install pyspark
```

## 🛠 使用说明

### 1. 数据准备
后期改进：将你的数据集(`租房数据.csv`)放入`data/`文件夹

### 2. 运行分析
- **数据探索与预处理**
  ```bash
  python scripts/数据探索.py
  ```
- **训练机器学习模型**
  ```bash
  python scripts/训练模型.py
  ```
- **对新数据进行预测**
  ```bash
  python scripts/预测.py --input data/新租房数据.csv
  ```

### 3. 查看结果
- **模型性能**：`results/模型评估.csv`
- **可视化图表**：`plots/价格分布.png`、`plots/特征重要性.png`
- **预测结果**：`output/预测结果.csv`

## 📊 示例结果
### 📈 租金价格分布
![价格分布](plots/价格分布.png)

### 🔍 特征重要性
![特征重要性](plots/特征重要性.png)

## 📝 未来改进方向
- [ ] 加入时间序列分析预测未来租金变化
- [ ] 整合NLP技术分析租房描述文本
- [ ] 开发Web交互界面(使用Flask/Dash)

## 🤝 贡献指南
欢迎提交Issue或Pull Request！

## 📜 许可证
MIT License
