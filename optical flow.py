import cv2
import numpy as np
import os
import pickle

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running",
    "walking"]

if __name__ == "__main__":

    # Create directory to store extracted SIFT features.
    os.makedirs("data", exist_ok=True)

    # Setup parameters for optical flow.
    farneback_params = dict(winsize = 20, iterations=1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
        pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)

    n_processed_files = 0

    for category in CATEGORIES:
        print("Processing category %s" % category)

        # Get all files in current category's folder.
        folder_path = os.path.join("..", "D:\\", category)
        filenames = os.listdir(folder_path)

        # List to store features. features[i] stores features for the i-th video
        # in current category.
        features = []

        for filename in filenames:
            filepath = os.path.join("..", "D:\\", category, filename)
            vid = cv2.VideoCapture(filepath)

            # Store features in current file.
            features_current_file = []

            prev_frame = None

            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break

                # Only care about gray scale.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None:
                    # Calculate optical flow.
                    flows = cv2.calcOpticalFlowFarneback(prev_frame, frame,
                        **farneback_params)

                    feature = []
                    for r in range(120):
                        if r % 10 != 0:
                            continue
                        for c in range(160):
                            if c % 10 != 0:
                                continue
                            feature.append(flows[r,c,0])
                            feature.append(flows[r,c,1])
                    feature = np.array(feature)

                    features_current_file.append(feature)

                prev_frame = frame

            features.append({
                "filename": filename,
                "category": category,
                "features": features_current_file 
            })

            n_processed_files += 1
            if n_processed_files % 30 == 0:
                print("Done %d files" % n_processed_files)

        pickle.dump(features, open("data/optflow_%s.p" % category, "wb"))


CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running",
    "walking"]

# Dataset are divided according to the instruction at:
# http://www.nada.kth.se/cvap/actions/00sequences.txt
TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
DEV_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

if __name__ == "__main__":

    train = []
    dev = []
    test = []

    # Store keypoints in training set.
    train_keypoints = []

    for category in CATEGORIES:
        print("Processing category %s" % category)
        category_features = pickle.load(open("data//optflow_%s.p" % category, "rb"))

        for video in category_features:
            person_id = int(video["filename"].split("_")[0][6:])

            if person_id in TRAIN_PEOPLE_ID:
                train.append(video)

                for frame in video["features"]:
                    train_keypoints.append(frame)

            elif person_id in DEV_PEOPLE_ID:
                dev.append(video)
            else:
                test.append(video)

print("Saving train/dev/test set to files")
pickle.dump(train, open("data/train.p", "wb"))
pickle.dump(dev, open("data/dev.p", "wb"))
pickle.dump(test, open("data/test.p", "wb"))
print("Saving keypoints in training set")
pickle.dump(train_keypoints, open("data/train_keypoints.p", "wb"))

import argparse

from sklearn.cluster import KMeans
from numpy import size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KMeans on training set")
    parser.add_argument("--dataset", type=str, default="data/train_keypoints.p",
                        help="number of clusters")
    parser.add_argument("--clusters", type=int, default=200,
                        help="number of clusters")

    args = parser.parse_args()
    dataset = args.dataset
    clusters = args.clusters

    print("Loading dataset")
    train_features = pickle.load(open(dataset, "rb"))
    n_features = len(train_features)

    print("Number of feature points to run clustering on: %d" % n_features)

    # Clustering with KMeans.
    print("Running KMeans clustering")
    kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10, n_jobs=2,
        verbose=1)
    kmeans.fit(train_features)

    # Save trained kmeans object to file.
pickle.dump(kmeans, open("data/cb_%dclusters.p" % clusters, "wb"))

from scipy.cluster.vq import vq

def make_bow(dataset, clusters, tfidf):
    print("Make bow vector for each frame")

    n_videos = len(dataset)

    bow = np.zeros((n_videos, clusters.shape[0]), dtype=np.float)

    # Make bow vectors for all videos.
    video_index = 0
    for video in dataset:
        visual_word_ids = vq(video["features"], clusters)[0]
        for word_id in visual_word_ids:
            bow[video_index, word_id] += 1
        video_index += 1

    # Check whether to use TF-IDF weighting.
    if tfidf:
        print("Applying TF-IDF weighting")
        freq = np.sum((bow > 0) * 1, axis = 0)
        idf = np.log((n_videos + 1) / (freq + 1))
        bow = bow * idf

    # Replace features in dataset with the bow vector we've computed.
    video_index = 0
    for i in range(len(dataset)):

        dataset[i]["features"] = bow[video_index]
        video_index += 1

        if (i + 1) % 50 == 0:
            print("Processed %d/%d videos" % (i + 1, len(dataset)))

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make bag of words vector")
    parser.add_argument("--codebook", type=str, default="data/cb_500clusters.p",
        help="path to codebook file")
    parser.add_argument("--tfidf", type=int, default=1,
        help="whether to use tfidf weighting")
    parser.add_argument("--dataset", type=str, default="data/train.p",
        help="path to dataset file")
    parser.add_argument("--output", type=str, default="data/train_bow_c500.p",
        help="path to output bow file")

    args = parser.parse_args()
    codebook_file = args.codebook
    tfidf = args.tfidf
    dataset = args.dataset
    output = args.output

    # Load clusters.
    codebook = pickle.load(open(codebook_file, "rb"))
    clusters = codebook.cluster_centers_

    # Load dataset.
    dataset = pickle.load(open(dataset, "rb"))

    # Make bow vectors.
    dataset_bow = make_bow(dataset, clusters, tfidf)

    # Save.
    pickle.dump(dataset_bow, open(output, "wb"))
    
from sklearn.ensemble import RandomForestClassifier


def make_dataset(dataset):
    X = []
    Y = []

    for video in dataset:
        X.append(video["features"])
        Y.append(video["category"])

    return X, Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVM on bow vectors")
    parser.add_argument("--dataset_bow", type=str, default="data/train_bow_c500.p",
        help="path to dataset bow file")
    parser.add_argument("--C", type=float, default=1)
    parser.add_argument("--output", type=str, default="data/svm_C1_c500.p")

    args = parser.parse_args()
    dataset_bow = args.dataset_bow
    C = args.C
    output = args.output

    # Load and make dataset.
    dataset = pickle.load(open(dataset_bow, "rb"))
    X, Y = make_dataset(dataset)

    # Train SVM and save to file.
    clf=RandomForestClassifier(n_estimators=500)
    clf.fit(X, Y)
    pickle.dump(clf, open(output, "wb")) 
    
CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", 
    "walking"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SVM classifier")
    parser.add_argument("--svm_file", type=str, default="data/svm_C1_c500.p",
        help="path to svm file")
    parser.add_argument("--bow_file", type=str, default="data/test_bow_c200.p",
        help="path to bow file")

    args = parser.parse_args()
    bow_file = args.bow_file
    svm_file = args.svm_file

    data = pickle.load(open(bow_file, "rb"))

    # Load trained SVM classifier.
    clf = pickle.load(open(svm_file, "rb"))

    confusion_matrix = np.zeros((6, 6))

    correct = 0
    for video in data:

        predicted = clf.predict([video["features"]])

        # Check if majority is correct.
        if predicted == video["category"]:
            correct += 1

        confusion_matrix[CATEGORIES.index(predicted),
            CATEGORIES.index(video["category"])] += 1

    print("%d/%d Correct" % (correct, len(data)))
    print("Accuracy =", correct / len(data))
    
    print("Confusion matrix")
    print(confusion_matrix)
    
    
    