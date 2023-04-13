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