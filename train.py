import seaborn as sns
import matplotlib.pyplot as plt
from dvclive import Live
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
y = iris.target


with Live() as live:
    live.log_param("epochs", 1)

    for c in [1, 1/10, 10]:
        plt.clf()
        clf = LogisticRegression(penalty='l2', C=c)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        live.log_metric('Precision', precision_score(y, y_pred, average='micro'))
        live.log_metric('Recall', recall_score(y, y_pred, average='micro'))
        conf_matrix = confusion_matrix(y, y_pred)
        sns_plot = sns.heatmap(conf_matrix, annot=True)
        results_path = 'results.png'
        plt.savefig(results_path)
        live.log_image(f"LogReg_C_param_is{c}.png", 'results.png')
        live.next_step()
