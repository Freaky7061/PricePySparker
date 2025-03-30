import os
import sys
from datetime import datetime
import pandas as pd

def setup_environment():
    # 创建必要的目录
    os.makedirs('./plot_data', exist_ok=True)
    os.makedirs('./plot_output', exist_ok=True)
    print("环境初始化完成...")

def run_data_processing():
    # 运行数据处理和模型训练
    try:
        from data_processing import process_data  
        print("开始数据处理和模型训练...")
        process_data()
        print("数据处理完成！")
        return True
    except Exception as e:
        print(f"数据处理出错: {str(e)}")
        return False

def run_visualization():
    # 运行可视化程序
    try:
        from plot_visualization import (
            plot_prediction_comparison, plot_residuals, plot_residuals_histogram,
            plot_area_price_analysis, plot_location_metro_analysis,
            plot_room_price_analysis, plot_community_analysis
        )
        print("开始生成可视化图表...")
        
        # 检查必要文件是否存在
        required_files = [
            './plot_data/prediction_results.csv',
            './plot_data/analysis_data.csv',
            './plot_data/r2_score.txt'
        ]
        for file in required_files:
            if not os.path.exists(file):
                print(f"错误：找不到数据文件 {file}！")
                return False
            
        # 生成预测相关图表
        plot_prediction_comparison()
        plot_residuals()
        plot_residuals_histogram()
        
        # 读取分析数据
        analysis_data = pd.read_csv('./plot_data/analysis_data.csv')
        
        # 生成分析图表
        plot_area_price_analysis(analysis_data)
        plot_location_metro_analysis(analysis_data)
        plot_room_price_analysis(analysis_data)
        plot_community_analysis(analysis_data)
        
        print("可视化图表生成完成！")
        return True
    except Exception as e:
        print(f"可视化生成出错: {str(e)}")
        return False

def main():

    start_time = datetime.now()
    print(f"程序开始运行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_environment()
    
    # 运行数据处理
    if not run_data_processing():
        print("数据处理失败，程序终止！")
        sys.exit(1)
    
    # 运行可视化
    if not run_visualization():
        print("可视化生成失败，程序终止！")
        sys.exit(1)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n程序运行完成！")
    print(f"总运行时间: {duration}")
    print(f"输出文件位置: ./plot_output/")

if __name__ == "__main__":
    main()
