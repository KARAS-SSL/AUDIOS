from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def create_svm_model():

    # Create a pipeline to standardize the data and then train the SVM
    clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear'))
    return clf