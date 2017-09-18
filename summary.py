import pickle
import numpy as np

def main():

    collect_data_file = open('collect.pkl','rb')
    collect_data = pickle.load(collect_data_file)
    collect_data_file.close()

    numUsers = len(collect_data['collect_train_users'])
    tweets = collect_data['collect_tweets']
    numTweets = len(tweets)

    cluster_data_file = open('cluster.pkl','rb')
    cluster_data = pickle.load(cluster_data_file )
    cluster_data_file.close()

    kmeans = cluster_data['cluster_kmeans']

    numCommunities = 2

    label_clusterZero_indices = np.where(kmeans.labels_==0)
    label_clusterOne_indices = np.where(kmeans.labels_==1)

    numUsers_CommZero =  len(label_clusterZero_indices[0])
    numUsers_CommOne = len(label_clusterOne_indices[0])
    avgNumUsers = (numUsers_CommZero + numUsers_CommOne)/2



    classify_data_file = open('classify.pkl','rb')
    classify_data = pickle.load(classify_data_file )
    classify_data_file.close()

    predictions = classify_data['classify_predictions']

    label_classZero_indices = np.where(predictions==0)
    label_classOne_indices = np.where(predictions==1)

    numUsers_ClassZero = len(label_classZero_indices[0])
    numUsers_ClassOne = len(label_classOne_indices[0])

    index_highestProb_zero = classify_data["classify_index_highestProb_zero"]
    index_highestProb_one = classify_data["classify_index_highestProb_one"]

    line1 = "Number of users collected: " + str(numUsers)
    line2 = "Number of messages collected: " + str(numTweets)
    line3 = "Number of communities discovered: " + str(numCommunities)
    line4 = "Number of users in community 0: " + str(numUsers_CommZero)
    line5 = "Number of users in community 1: " + str(numUsers_CommOne)
    line6 = "Average number of users per community: " + str(avgNumUsers)
    line7 = "Number of users in class 0: " + str(numUsers_ClassZero)
    line8 = "Number of users in class 1: " + str(numUsers_ClassOne)
    line9 = "Instance of class 0: " + tweets[index_highestProb_zero]['text'] + ": : User: " + tweets[index_highestProb_zero]['user']['name']
    line10 = "Instance of class 1: " + tweets[index_highestProb_one]['text'] + ": : User: " + tweets[index_highestProb_one]['user']['name']

    lines = [line1, line2, line3, line4, line5, line6, line7, line8, line9, line10]


    f = open("summary.txt", "w")
    for line in lines:
        f.write(line)
        f.write("\n")
    f.close()


if __name__ == main():
    main()
