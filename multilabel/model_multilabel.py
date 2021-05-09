import pickle


def train_predict_save(clf, clf_name, X_train, Y_train, X_test, Y_test, dest_dir="data"):

    print("\nFitting: ", clf_name)

    # fit
    clf.fit(X_train, Y_train)

    # predict train and test set
    Y_train_pred = clf.predict(X_train)
    Y_test_pred = clf.predict(X_test)

    # save fitted model
    with open(dest_dir + "/" + clf_name + "_clf.pkl", "wb") as f:
        pickle.dump(clf, f, protocol=4)
    
    # save predictions
    with open(dest_dir + "/" + clf_name + "_y_train_pred.pkl", "wb") as f:
        pickle.dump(Y_train_pred, f, protocol=4)
        
    with open(dest_dir + "/" + clf_name + "_y_test_pred.pkl", "wb") as f:
        pickle.dump(Y_test_pred, f, protocol=4)
