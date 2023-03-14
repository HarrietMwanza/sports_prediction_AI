from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# Q1:we import only files relevant to what we want to analyse
# Load the rankings file
# we add an extra column called "player_rating to the file"


def runner_function():
    # Load the dataset
    df = pd.read_csv(r'./rankings_1973-2017.csv')

    # Create a new column called "player_rating"
    df['player_rating'] = (df.groupby('player_id')['ranking_points'].rolling(window=52).mean()  # Average ranking points over the past year
                           .reset_index(drop=True))  # Reset the index to remove the multi-level index from the groupby operation

    # Calculate the number of titles and runner-up finishes for each player
    titles = df[df['move_positions'] == 1].groupby(
        'player_id').size().reset_index(name='num_titles')
    runners_up = df[df['move_positions'] == 2].groupby(
        'player_id').size().reset_index(name='num_runners_up')
    df = pd.merge(df, titles, on='player_id', how='left')
    df = pd.merge(df, runners_up, on='player_id', how='left')

    # Fill missing values with 0
    df['num_titles'].fillna(0, inplace=True)
    df['num_runners_up'].fillna(0, inplace=True)

    # Normalize the features
    df['player_rating'] = (df['player_rating'] - df['player_rating'].min()) / \
        (df['player_rating'].max() - df['player_rating'].min())
    df['num_titles'] = (df['num_titles'] - df['num_titles'].min()) / \
        (df['num_titles'].max() - df['num_titles'].min())
    df['num_runners_up'] = (df['num_runners_up'] - df['num_runners_up'].min()) / \
        (df['num_runners_up'].max() - df['num_runners_up'].min())

    # Print the first 10 rows of the updated dataset
    print(df.head(10))

    # Check the shape and column names of the rankings dataframe
    print(df.shape)

    # Check the shape and column names of the tournaments dataframe
    print(df.columns)

    # Check for missing values
    print(df.isnull().sum())
    # Drop the rows with missing values
    df.dropna(inplace=True)

    print("Q1 done")

    # Q2:Create feature subsets which show maximum correlation with the dependent variable. [5]

    # Select the relevant features
    features = ['move_positions', 'tourneys_played',
                'player_age', 'ranking_points']
    # One-hot encode the move_positions column
    move_positions_encoded = pd.get_dummies(
        df['move_positions'], prefix='move_positions')

    print("Done with encoding move_positions")

    # One-hot encode the move_direction column
    move_direction_encoded = pd.get_dummies(
        df['move_direction'], prefix='move_direction')

    print("Done with encoding move_direction")

# Concatenate the encoded columns with the original data frame: TODO: THIS IS NOT WORKING - just hangs in there
    df_encoded = pd.concat(
        [df, move_positions_encoded, move_direction_encoded], axis=1)

    print("Done with concatenating")

    # Drop the original move_positions and move_direction columns
    df_encoded.drop(['move_positions', 'move_direction'], axis=1, inplace=True)

    print("Done with encoding")

    # Print the first few rows of the encoded data frame to check the result
    print(df_encoded.head())

    # Select the variables we want to include in the correlation matrix to see correclation to player ratings
    cols = ['move_positions', 'tourneys_played',
            'player_age', 'move_direction', 'ranking_points']

    # Subset the data frame to only include those columns
    subset = df[cols]

    # Compute the correlation matrix
    corr_matrix = subset.corr()

    # Print the matrix
    print(corr_matrix)

    print("Q2 done")

    # Q3:Create and train a suitable machine learning model that can predict a player rating. [5]
    # Load the data

    # Select the relevant columns

    # Drop missing values
    subset.dropna(inplace=True)

    # Convert categorical variables to numeric using one-hot encoding
    subset_encoded = pd.get_dummies(
        subset, columns=['move_positions', 'move_direction'])

    # Split the data into training and testing sets
    X = subset_encoded.drop('player_rating', axis=1)
    y = subset_encoded['player_rating']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("Q3 done")

    # Q4:Measure the performance of the model and fine tune it as a process of optimization. [5]
    # Load the data

    # Convert categorical variables to numeric using one-hot encoding
    # Create a linear regression model
    lr = LinearRegression()

    # Train the model
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr.predict(X_test)

    # Evaluate the performance using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error:", mse)

    # Fine-tune the model
    # You can fine-tune the model by adjusting its hyperparameters or by using a different algorithm altogether.
    # For example, you can try using a different regression algorithm like Ridge or Lasso, or you can adjust the regularization strength.

    # Here's an example of how you can fine-tune the hyperparameters of a Ridge regression model using grid search:

    # Define the hyperparameters to search over
    param_grid = {'alpha': [0.1, 1, 10]}

    # Create a Ridge regression model
    ridge = Ridge()

    # Use GridSearchCV to search over the hyperparameters
    grid_search = GridSearchCV(ridge, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found
    print("Best hyperparameters:", grid_search.best_params_)

    # Train a new Ridge regression model with the best hyperparameters found
    ridge = Ridge(alpha=grid_search.best_params_['alpha'])
    ridge.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = ridge.predict(X_test)

    # Evaluate the performance using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error after fine-tuning:", mse)

    # Q5:Use the data from another season which was not used during the training to test how good the model is. [5]
    # new dataset used is found here :https://www.kaggle.com/datasets/paritosh712/wta-tennis-rankings-data

    # load the new dataset into a pandas DataFrame
    test_data = pd.read_csv(r'./WTAdata.csv')
    print(df)

    # preprocess the test data in the same way as the training data
    test_data['date'] = pd.to_datetime(test_data['date'])
    test_data = test_data.dropna(subset=['rank_points'])
    test_data = test_data[['player_id', 'date', 'rank_points']]
    test_data = test_data.sort_values(['player_id', 'date'])
    test_data['previous_rank_points'] = test_data.groupby('player_id')[
        'rank_points'].shift(1)
    test_data['ranking_change'] = test_data['previous_rank_points'] - \
        test_data['rank_points']
    test_data = test_data.dropna()

    # use the trained model to make predictions on the test data
    test_X = test_data[['previous_rank_points', 'ranking_change']].values
    test_y = test_data['rank_points'].values
    test_predictions = lr.predict(test_X)

    # evaluate the performance of the model on the test data
    mse = mean_squared_error(test_y, test_predictions)
    print("MSE:", mse)

    return lr

    # Q6:Deploy the model on a simple web page using either (Heroku, Streamlite or Flask)
    # and upload a link to the video that shows how the model performs on the web page/site. [5]


# function to be used by flask

def predict_rating(previous_rank_points, ranking_change):

    lr = runner_function()

    # create a numpy array from the input parameters
    input_data = np.array([[previous_rank_points, ranking_change]])

    # get the prediction
    prediction = lr.predict(input_data)

    # return the prediction
    return prediction[0]
