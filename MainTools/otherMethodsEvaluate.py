import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def randomForestPredict(dataframePath, aimColumn, test_size=0.3, random_state=42):
    # 使用随机森林进行预测,输入dataframe和aimcolumn,输出预测结果

    # 分离特征和目标变量,使用aimColumn作为目标变量
    dataframe = pd.read_csv(dataframePath)
    if aimColumn not in dataframe.columns:
        raise ValueError(f"指定的 aimColumn '{aimColumn}' 不在 DataFrame 的列中。")

    y = dataframe[aimColumn].values
    X = dataframe.drop(columns=[aimColumn]).values

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 创建随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    # 训练模型
    clf.fit(X_train, y_train)
    
    # 预测测试集
    y_pred = clf.predict(X_test)
    
    # 计算F1 P R
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return metrics


def decisionTreePredict(dataframePath, aimColumn, test_size=0.3, random_state=42):
    # 使用决策树进行预测,输入dataframe和aimcolumn,输出预测结果
    dataframe = pd.read_csv(dataframePath)
    if aimColumn not in dataframe.columns:
        raise ValueError(f"指定的 aimColumn '{aimColumn}' 不在 DataFrame 的列中。")

    y = dataframe[aimColumn].values
    X = dataframe.drop(columns=[aimColumn]).values

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 创建决策树分类器
    clf = DecisionTreeClassifier(random_state=random_state)

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测测试集
    y_pred = clf.predict(X_test)
    
    # 计算F1 P R
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return metrics

def knnPredict(dataframePath, aimColumn, test_size=0.3, random_state=42):
    # 使用KNN进行预测,输入dataframe和aimcolumn,输出预测结果
    dataframe = pd.read_csv(dataframePath)
    if aimColumn not in dataframe.columns:
        raise ValueError(f"指定的 aimColumn '{aimColumn}' 不在 DataFrame 的列中。")

    y = dataframe[aimColumn].values
    X = dataframe.drop(columns=[aimColumn]).values

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 创建KNN分类器
    clf = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2)

    # 训练模型
    clf.fit(X_train, y_train)
    
    # 预测测试集
    y_pred = clf.predict(X_test)
    
    # 计算F1 P R
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return metrics

def svmPredict(dataframePath, aimColumn, test_size=0.3, random_state=42):
    # 使用SVM进行预测,输入dataframe和aimcolumn,输出预测结果
    dataframe = pd.read_csv(dataframePath)
    if aimColumn not in dataframe.columns:
        raise ValueError(f"指定的 aimColumn '{aimColumn}' 不在 DataFrame 的列中。")

    y = dataframe[aimColumn].values
    X = dataframe.drop(columns=[aimColumn]).values

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 创建SVM分类器
    clf = SVC(kernel='linear', random_state=random_state)

    # 训练模型
    clf.fit(X_train, y_train)
    
    # 预测测试集
    y_pred = clf.predict(X_test)
    
    # 计算F1 P R
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return metrics
    
