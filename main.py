import pandas as pd
import numpy as np
import seaborn as sb
import praw
import datetime as dt
import sklearn.model_selection
from sklearn import preprocessing
from matplotlib import pyplot as plt
from wordcounter.wordcounter import WordCounter


def get_date(created):
    return dt.datetime.fromtimestamp(created)


def word_count(in_title):
    count = []
    for j in range(len(title)):
        word_counter = WordCounter(in_title[j], delimiter=' ')
        count.append(word_counter.get_word_count())
    return count


def hypothesis_cal(Theta, x):
    hyp = x @ Theta
    return hyp


def error_cal(Theta, x, y):
    error = np.transpose(x @ Theta - y)
    return error


def cost_cal(Theta, m, x, y):
    cost = (1 / (2 * m)) * np.transpose((x @ Theta - y)) @ (x @ Theta - y)
    return cost


if __name__ == '__main__':
    reddit = praw.Reddit(client_id='nVuO8hNgCZb6Vw',
                         client_secret='l8CvvaH76odKbqV1GoR1-B8cbAg',
                         user_agent='GradientDescent w/ reddit',
                         username='SatisfactionVisual91',
                         password='Thinh123')

    subreddit = reddit.subreddit('AskReddit')
    top_subreddit = subreddit.top(limit=1000)

    topics_dict = {"title": [],
                   "karma": [],
                   "number_of_comments": [],
                   "created": [],
                   "body": []}

    for submission in top_subreddit:
        topics_dict["title"].append(submission.title)
        topics_dict["karma"].append(submission.score)
        topics_dict["number_of_comments"].append(submission.num_comments)
        topics_dict["created"].append(submission.created)
        topics_dict["body"].append(submission.selftext)

    data = pd.DataFrame(topics_dict)
    time = data["created"].apply(get_date)
    data = data.assign(timestamp=time)
    data.to_csv("Reddit_data.csv", index=False)

    data = pd.read_csv("Reddit_data.csv", sep=",")
    data = data[["title", "karma", "number_of_comments"]]
    title = data["title"]

    data = data.assign(title_count=word_count(title))
    data = data[["title_count", "karma", "number_of_comments"]]
    # Graph data
    cols = data.columns
    sb.pairplot(data[cols], height=2.5)
    plt.tight_layout()

    # Splitting the data
    Y_Predict = "number_of_comments"
    Xs = np.array(data.drop([Y_Predict], 1))
    Y = np.array(data["number_of_comments"])

    # Splitting train data and test data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        Xs, Y, test_size=0.2)

    # Scaled data
    x_scaled = preprocessing.scale(x_train)
    y_scaled = preprocessing.scale(y_train)
    x_test_scaled = preprocessing.scale(x_test)
    y_test_scaled = preprocessing.scale(y_test)

    # Number of features
    n = x_scaled.shape[1]

    # Size of train data
    size = len(x_scaled)

    # Initialize theta
    theta = np.ones(n)

    # Hypothesis
    hypothesis = hypothesis_cal(theta, x_scaled)

    # Cost
    cost_val = cost_cal(theta, size, x_scaled, y_scaled)

    # Gradient Descent
    learning_rate = 0.001
    epoch = 1000
    loss = error_cal(theta, x_scaled, y_scaled)

    for i in range(epoch):
        # Gradient Descent
        theta = theta - learning_rate * \
            (1 / size) * np.transpose(x_scaled) @ (x_scaled @ theta - y_scaled)

    # Hypothesis
    hypothesis = hypothesis_cal(theta, x_scaled)

    # Cost
    cost_val = cost_cal(theta, size, x_scaled, y_scaled)

    # Loss
    loss = error_cal(theta, x_scaled, y_scaled)

    print("-----Train data------")
    for i in range(size):
        print([i], "Real number of comments: ", round(y_scaled[i], 5),
              "  Predicted number of comments: ", round(hypothesis[i], 5),
              " Error: ", round(loss[i], 5))

    print("Cost: ", cost_val)
    print("Co-ef: ", theta, "\n")
    plt.show()
