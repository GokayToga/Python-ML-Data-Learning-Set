import pandas as pd
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']},index=['Product A', 'Product B'])

pd.Series([1, 2, 3, 4, 5])
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
reviews.shape
reviews.head()
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()

animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals.to_csv('cows_and_goats.csv')

#Hence to access the country property of reviews we can use
reviews.country
reviews['country']
#It pretty much is, so it's no surprise that, to drill down to a single specific value,
#we need only use the indexing operator [] once more:
reviews['country'][0]
#To select the first row of data in a DataFrame, we may use the following
reviews.iloc[0]
#Both loc and iloc are row-first, column-second. This is the opposite of what we do in native Python,
#which is column-first, row-second.
#To get a column with iloc, we can do the following
reviews.iloc[:, 0]
#to select the country column from just the first, second, and third row, we would do
reviews.iloc[:3, 0]
#to select just the second and third entries, we would do
reviews.iloc[1:3, 0]
#It's also possible to pass a list
reviews.iloc[[0, 1, 2], 0]
#it can also work negative values, which will count from the end of the values
reviews.iloc[-5:]
#loc, by contrast, uses the information in the indeices to do its work.
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]

#iloc uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded.
#So 0:10 will select entries 0,...,9. loc, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10


#We can manipulate the index in any way we see fit.
#For example, we can set the index to be the title of each wine.
reviews.set_index("title")

#For example, suppose that we're interested specifically in better-than-average wines produced in Italy.
#We can start by checking if each wine is Italian or not:
reviews.country == 'Italy'
#We can then use that selection to filter the reviews down to just those of Italian wines:
reviews.loc[reviews.country == 'Italy']
#We can use the ampersand (&) to bring the two questions together so we get the reviews down to just those of Italian wines that scored above average:
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
#any wine that's made in Italy or which is rated above average. For this we use a pipe (|):
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
#isin is lets you select data whose value "is in" a list of values.
reviews.loc[reviews.country.isin(['Italy', 'France'])]
#isnull and notnull let you highlight values which are (or are not) empty (NaN in numeric columns, None or NaN in object columns)
reviews.loc[reviews.price.notnull()]

#Assigning data
#Assigning constant value
reviews['critic'] = 'everyone'
reviews['critic']
#Assigning itterable values
reviews['index_backwards'] = range(len(reviews), 0, -1)
reviews['index_backwards']

#median of the pont column in reviews
median_points =reviews.points.median()
#countries represented
countries = reviews.country.unique()
#count of reviews of each country
reviews_per_country = reviews.country.value_counts()
#removing the mean from price
centered_price = reviews.price - reviews.price.mean()
#best price to points ratio
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']
#counting tropical and fruity
n_trop = reviews.description.map(lambda desc: 'tropical' in desc).sum()
n_fruity = reviews.description.map(lambda desc: 'fruity' in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity']) 

#a Series whose index is the taster_twitter_handle category from the dataset, and whose values count how many reviews each person wrote.
reviews_written = reviews.groupby('taster_twitter_handle').size()
#a Series whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. 
# Sort the values by price, ascending (so that 4.0 dollars is at the top and 3300.0 dollars is at the bottom).
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()
#Create a DataFrame whose index is the variety category from the dataset and whose values are the min and max values thereof.
price_extremes = reviews.groupby('variety').price.agg([min, max])
#Create a variable sorted_varieties containing a copy of the dataframe 
#from the previous question where varieties are sorted in descending order based on minimum price, then on maximum price (to break ties).
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)
#Create a Series whose index is reviewers and whose values is the average review score given out by that reviewer. Hint: you will need the taster_name and points columns.
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
#Create a Series whose index is a MultiIndexof {country, variety} pairs. 
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)

#data type of the points column in the datase
dtype = reviews.points.dtype
#Create a Series from entries in the points column, but convert the entries to strings. Hint: strings are str in native Python.
point_strings = reviews.points.astype(str)
#Sometimes the price column is null. How many reviews in the dataset are missing a price?
n_missing_prices = reviews.price.isnull().sum()
#Create a Series counting the number of times each value occurs in the region_1 field. This field is often missing data, so replace missing values with Unknown.
reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False) 

#region_1 and region_2 are pretty uninformative names for locale columns in the dataset. Create a copy of reviews with these columns renamed to region and locale, respectively.
renamed = reviews.rename(columns={'region_1': 'region', 'region_2': 'locale'})
#Set the index name in the dataset to wines.
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
#Create a DataFrame of products mentioned on either subreddit
combined_products = pd.concat([gaming_products, movie_products])
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
#Both tables include references to a MeetID, a unique key for each meet (competition) included in the database. Using this, generate a dataset combining the two tables into one