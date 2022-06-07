import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder


def make_baseline(df: pd.DataFrame):
    """
    Calculate the baseline dataframe based on user interaction.
    :param df: pandas dataframe
    :return: pandas dataframe: ordered baseline dataframe
    """
    baseline_df = df.copy()
    # make a score by mapping the events with the point system
    baseline_df['score'] = baseline_df['event_type'].map(
        {'view': 1, 'cart': 10, 'purchase': 25, "remove_from_cart": -5})
    # count the times that a product is purchased
    baseline_df['times_purchased'] = baseline_df['event_type'].apply(lambda x: 1 if x == 'purchase' else 0)
    # group the dataframe by the products_id
    baseline_df = baseline_df.groupby('product_id', as_index=False, sort=False).agg(
        {'score': 'sum', 'times_purchased': 'sum', 'product_id': 'first', 'category_id': 'first'})
    # order the dataframe by the score
    baseline_df = baseline_df.sort_values(by=['score'], ascending=False)

    return baseline_df


def baseline(df: pd.DataFrame):
    """
    Calculate and write the baseline.
    The baseline is based on a scoring system that scores a product based on user interactions.
    :param df: pandas dataframe
    :return: list of product_ids: the baseline
    """
    # make the baseline
    baseline_df = make_baseline(df)
    # only keep the top 10 scoring products
    top10_df = baseline_df.head(10)

    # write the baseline.csv
    top10_df["product_id"].to_csv("baseline.csv", header=False, index=False)

    # get the categories of the top 10
    top10_categories = top10_df["category_id"].unique()
    top10_products = top10_df["product_id"].tolist()

    # dataframe to calculate the upper limit by counting sales in categories of top 10 products
    upper_limit_df = baseline_df.groupby('category_id', as_index=False, sort=False).agg(
        {"times_purchased": "sum", "category_id": "first"})
    category_sales = 0
    category_users = []

    # Go over all the categories of top 10 products
    for category in top10_categories:
        # get users that bought from category
        category_users_df = df[df["category_id"] == category]
        category_users += category_users_df[category_users_df["event_type"] == "purchase"]['user_id'].unique().tolist()
        # make sure all items in the list are unique
        category_users = set(category_users)
        category_users = list(category_users)

        # get category sales
        category_sales += (upper_limit_df[upper_limit_df["category_id"] == category].iloc[0]["times_purchased"])

    # get users that bought top 10 products
    users = []
    for product in top10_products:
        users_df = df[df["product_id"] == product]
        users += users_df[users_df["event_type"] == "purchase"]['user_id'].unique().tolist()
        # make sure all items in the list are unique
        users = set(users)
        users = list(users)

    # print the results
    print("Sales:")
    print(baseline_df["times_purchased"].sum(), " total sales")
    print(category_sales, " total category sales")
    print(top10_df["times_purchased"].sum(), " top 10 purchases")
    print("Users:")
    print(df["user_id"].nunique(), " all users")
    print(len(category_users), "users that bought top 10 category")
    print(len(users), "users that bought top 10 product")

    return top10_df["product_id"].tolist()


def make_rules(df: pd.DataFrame):
    """
    Make association rules with fpgrowth based on the dataset.
    :param df: pandas dataframe
    :return: pandas dataframe: the association rules
    """
    # filter the remove_from_cart event
    transactions_df = df[df["event_type"].isin(["purchase", "view", "cart"])]
    # group the dataframe by the users
    transactions_df = transactions_df.groupby("user_id")["product_id"].apply(set).reset_index()

    # make and fit the encoder
    encoder = TransactionEncoder()
    encoder.fit(transactions_df['product_id'])

    # encode the dataframe
    encoded_df = encoder.transform(transactions_df['product_id'])
    encoded_df = pd.DataFrame(encoded_df, columns=encoder.columns_)

    # make the rules
    frequent_itemset = fpgrowth(encoded_df, min_support=0.00005, use_colnames=True, max_len=2)
    rules = association_rules(frequent_itemset, metric="confidence", min_threshold=0.1)
    rules = rules.sort_values("lift", ascending=False)
    # unfreeze consequents
    rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
    return rules


def recommendations(rules_df: pd.DataFrame, top10_list: list, recommend_df: pd.DataFrame, write_bool: bool = True):
    """
    Make personal recommendations based on previous user interactions given the rules.
    Recommendations are made with association rules.
    :param rules_df: pandas dataframe with the rules
    :param top10_list: list with baseline
    :param recommend_df: pandas dataframe with previous user interactions
    :param write_bool: boolean whether to write solutions
    :return: pandas dataframe with recommendations for each user
    """
    # the dataframe where the recommendations will be stored
    recommendation_df = pd.DataFrame(columns=(
        'user_id', 'product_id 1', 'product_id 2', 'product_id 3', 'product_id 4', 'product_id 5', 'product_id 6',
        'product_id 7', 'product_id 8', 'product_id 9', 'product_id 10'))
    # filter the remove_from_cart event
    recommender_df = recommend_df[recommend_df["event_type"].isin(["purchase", "view", "cart"])]
    # list all products a user interacted with for each user
    user_df = recommender_df.groupby('user_id')['product_id'].apply(list).reset_index()
    # a list of all users
    user_list = recommender_df['user_id'].unique()

    # go over all users
    for user_index in range(len(user_list)):
        user = user_list[user_index]
        # get all rules of the products where the user interacted with
        product_rules = user_df[user_df.user_id == user]["product_id"].tolist()[0]
        # make every item in this list unique
        product_rules = set(product_rules)
        product_rules = list(product_rules)

        predictions = []
        # get all the information from these rules: items they point to with the lift and confidence
        for product_rule in product_rules:
            prediction = rules_df[rules_df['antecedents'] == {product_rule}][["consequents", "lift", "confidence"]]
            predictions += prediction.values.tolist()

        item_list = []
        score_list = []
        # go over all the rules(predictions) and get the items they point to and the lift in a list
        for item, score, _ in predictions:
            # if the lift is high enough and the item is not yet in the list add it and the lift
            if item not in item_list and score > 100:
                item_list.append(item)
                score_list.append(score)
            # if the item is already in the list add the current lift to the score
            elif item in item_list:
                index = item_list.index(item)
                score_list[index] += score
        # list for the user recommendation, initialize with the user id
        user_recommendation = [user_list[user_index]]
        # sort the products by the lift
        user_recommendation += [x for _, x in sorted(zip(score_list, item_list), reverse=True)]

        # if there are to many recommendations only keep the top 10 (+ user id)
        if len(user_recommendation) > 11:
            user_recommendation = user_recommendation[:11]
        # if there are to little recommendation fill it with the baseline
        elif len(user_recommendation) < 11:
            top_index = 0
            # make sure the baseline does not add duplicate products
            while len(user_recommendation) < 11 and top_index < 10:
                if str(top10_list[top_index]) not in user_recommendation:
                    user_recommendation.append(str(top10_list[top_index]))
                top_index += 1

        # add the recommendation to the dataframe
        recommendation_df.loc[user_index] = user_recommendation
    # save the recommendations to a csv file
    if write_bool:
        recommendation_df.to_csv("recommendations.csv", header=False, index=False)
    return recommendation_df


def test_recommendations(rules_df, top10_list, recommend_df):
    """
    Test the personalized recommendations against the baseline.
    :param rules_df: pandas dataframe with the association rules
    :param top10_list: list of baseline
    :param recommend_df: pandas dataframe with previous user interactions
    """
    # key for splitting the data completely
    recommend_df["Key"] = recommend_df.product_id.astype("string") + "_" + recommend_df.user_id.astype("string")

    # split the data by solutions which are the purchased items and past_data which are the other events
    past_data = recommend_df[recommend_df["event_type"].isin(["view", "cart"])]
    solutions = recommend_df[recommend_df["event_type"] == "purchase"]

    # uncomment to split the data completely: If A is purchased in solutions than it can't be viewed in past_data by that user
    # past_data = past_data[~(past_data.Key.isin(solutions.Key))]

    # make the recommendations with the past_data
    recommendation_df = recommendations(rules_df, top10_list, past_data, False)

    # get the users of both dataframes
    recommended_user = recommendation_df["user_id"].tolist()
    solution_user = solutions["user_id"].tolist()

    predicted = 0
    not_predicted = 0
    # go over all the users that are in both dataframes
    for user in recommended_user:
        if user in solution_user:
            # get the products in the solution
            products = solutions[solutions.user_id == user]["product_id"].unique()
            # go over all these products
            for product in products:
                # check if the personalized recommender recommended it
                if str(product) in recommendation_df[recommendation_df.user_id == user].values:
                    predicted += 1
                else:
                    not_predicted += 1

    baseline_predicted=0
    baseline_not_predicted=0
    # go over all the users that are in both dataframes
    for user in recommended_user:
        if user in solution_user:
            # get the products in the solution
            products = solutions[solutions.user_id == user]["product_id"].unique()
            for product in products:
                # check if the product is predicted by the baseline
                if product in top10_list:
                    baseline_predicted += 1
                else:
                    baseline_not_predicted += 1

    # print results
    print("The baseline recommender predicted", baseline_predicted, "items and failed to predict", baseline_not_predicted)
    print("The personalized recommender predicted", predicted, "items and failed to predict",  not_predicted)


if __name__ == '__main__':
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', desired_width)

    dataset = pd.read_csv("dataset.csv")
    test = pd.read_csv("recommend.csv")

    # make the baseline
    print("BASELINE")
    top10 = baseline(dataset)

    # make the rules
    rules_dataframe = make_rules(dataset)

    # make recommendations
    recommendation_df = recommendations(rules_dataframe, top10, test)

    # test personalized recommendations against the baseline
    print("\nTESTS")
    test_recommendations(rules_dataframe, top10, test)
