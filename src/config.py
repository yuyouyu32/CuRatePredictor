import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Classificaiton config
RegDataPath = '../data/data.xlsx'  # Replace with your file path
RegResultpath = '../results/'
# FeatureColumns = ['处理量/t', '粗选浓度/%', '粗选-200目', '精选-400目', '原矿', '粗选精矿', '黄药g/t', '松油g/t', '精选精矿'] # with R
# DropColumns = ['磨矿浓度/%', '粗选尾矿', '精选尾矿', '粗选', '精选']
FeatureColumns = ['处理量/t', '粗选浓度/%', '粗选-200目', '精选-400目', '原矿', '黄药g/t', '松油g/t']   # w/o R
DropColumns = ['磨矿浓度/%', '粗选尾矿', '精选尾矿', '粗选', '精选', '粗选精矿', '精选精矿']
ProcessColumnsEn = \
{
    '处理量/t': 'ProcessAmount',
    '磨矿浓度/%': 'GrindingConcentration',
    '粗选浓度/%': 'RoughConcentration',
    '粗选-200目': 'Rough-200Mesh',
    '精选-400目': 'Fine-400Mesh',
    '原矿': 'RawOre',
    '粗选精矿': 'RoughConcentrate',
    '粗选尾矿': 'RoughTailings',
    '黄药g/t': 'YellowMedicine',
    '松油g/t': 'PineOil',
    '精选精矿': 'FineConcentrate',
    '精选尾矿': 'FineTailings'
}
LabelsColumns = ['综合']

process_method = 'Single'  # 'Single' or 'Multi'





