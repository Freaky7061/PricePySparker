import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

# 读取数据
plot_data = pd.read_csv('./plot_data/prediction_results.csv')
with open('./plot_data/r2_score.txt', 'r') as f:
    r2_score = float(f.read())

def plot_prediction_comparison():
    plt.figure(figsize=(10, 6))
    plt.scatter(plot_data['actual_price'], 
               plot_data['predicted_price'], 
               alpha=0.5)
    
    # 添加对角线
    min_val = min(plot_data['actual_price'].min(), 
                 plot_data['predicted_price'].min())
    max_val = max(plot_data['actual_price'].max(), 
                 plot_data['predicted_price'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', lw=2)

    plt.xlabel('实际房价（元）')
    plt.ylabel('预测房价（元）')
    plt.title('线性回归模型预测结果对比')
    
    # 添加R²分数
    plt.text(0.05, 0.95, f'R方 = {r2_score:.3f}', 
             transform=plt.gca().transAxes, 
             fontsize=12)
    
    plt.tight_layout()
    plt.savefig('./plot_output/prediction_comparison.png', dpi=300)
    plt.close()

def plot_residuals():
    # 绘制残差散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(plot_data['predicted_price'], 
               plot_data['residuals'], 
               alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel('预测房价（元）')
    plt.ylabel('残差（元）')
    plt.title('残差分布图')
    
    plt.tight_layout()
    plt.savefig('./plot_output/residuals_scatter.png', dpi=300)
    plt.close()

def plot_residuals_histogram():
    # 绘制残差分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(plot_data['residuals'], bins=50, alpha=0.75)
    
    plt.xlabel('残差（元）')
    plt.ylabel('频数')
    plt.title('残差分布直方图')
    
    plt.tight_layout()
    plt.savefig('./plot_output/residuals_histogram.png', dpi=300)
    plt.close()

def plot_area_price_analysis(df):
    # 绘制面积与价格关系的分析图
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.scatterplot(data=df, x='area', y='price', alpha=0.5)
    plt.title('面积与价格散点图')
    plt.xlabel('面积(平方米)')
    plt.ylabel('价格(元/月)')
    
    plt.subplot(122)
    sns.kdeplot(data=df, x='area', y='price', cmap='YlOrRd')
    plt.title('面积-价格热力分布图')
    plt.xlabel('面积(平方米)')
    plt.ylabel('价格(元/月)')
    
    plt.tight_layout()
    plt.savefig('./plot_output/area_price_relation.png', dpi=300)
    plt.close()

def plot_location_metro_analysis(df):
    # 绘制位置与地铁分析图
    plt.figure(figsize=(15, 6))
    location_avg_price = df.groupby('pos1')['price'].mean().sort_values(ascending=False)
    
    plt.subplot(121)
    sns.boxplot(data=df, x='pos1', y='price', order=location_avg_price.index)
    plt.xticks(rotation=45)
    plt.title('各区域房价分布')
    plt.xlabel('区域')
    plt.ylabel('价格(元/月)')
    
    plt.subplot(122)
    df['是否临近地铁'] = df['subway'].notna()
    sns.boxplot(data=df, x='pos1', y='price', hue='是否临近地铁', 
                order=location_avg_price.index[:8])
    plt.xticks(rotation=45)
    plt.title('地铁对各区域房价的影响')
    plt.xlabel('区域')
    plt.ylabel('价格(元/月)')
    
    plt.tight_layout()
    plt.savefig('./plot_output/location_metro_analysis.png', dpi=300)
    plt.close()

def plot_room_price_analysis(df):
    # 绘制户型分析图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    sns.violinplot(data=df, x='bedrooms', y='price')
    plt.title('房间数与价格分布')
    plt.xlabel('房间数量')
    plt.ylabel('价格(元/月)')
    
    plt.subplot(122)
    sns.boxplot(data=df, x='bedrooms', y='unit_price')
    plt.title('房间数与单位面积价格分布')
    plt.xlabel('房间数量')
    plt.ylabel('单位面积价格(元/平米/月)')
    
    plt.tight_layout()
    plt.savefig('./plot_output/room_price_analysis.png', dpi=300)
    plt.close()

def plot_community_analysis(df):
    # 绘制小区分析图
    plt.figure(figsize=(12, 6))
    community_stats = df.groupby('community').agg({
        'price': ['mean', 'count']
    }).reset_index()
    community_stats.columns = ['Community', 'Average Price', 'House Number']
    top_communities = community_stats.nlargest(15, 'House Number')
    
    plt.bar(range(len(top_communities)), top_communities['Average Price'])
    plt.xticks(range(len(top_communities)), top_communities['Community'], 
               rotation=45, ha='right')
    plt.title('热门小区平均房价对比')
    plt.xlabel('小区名称')
    plt.ylabel('平均价格(元/月)')
    
    for i, v in enumerate(top_communities['House Number']):
        plt.text(i, top_communities['Average Price'].iloc[i], 
                f'House Number: {v}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('./plot_output/community_analysis.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    # 只保留创建输出目录
    import os
    os.makedirs('./plot_output', exist_ok=True)
    print("输出目录已创建！") 