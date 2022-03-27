import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing

covid = pandas.read_csv('covid-19-data/us-counties-recent.csv', sep=',')
covid = covid.get(['fips', 'cases', 'deaths'])

population = pandas.read_csv('covid-19-data/county_complete.csv', sep=',').get(
    ['fips', 'pop2017', 'median_household_income_2017'])


covid = covid.dropna()
covid.set_index("fips")
covid['fips'] = covid['fips'].astype(int)
population = population.dropna()
population.set_index("fips")
min_max_scaler = preprocessing.MinMaxScaler()

joined = covid.merge(population, on='fips')


joined['deaths-corrected'] = joined['deaths'] / joined['pop2017'] * 100000
joined['death-cases'] = joined['deaths'] / joined['cases']
print(joined[0:1])
joined = joined.loc[(joined['deaths'] != 0)]
# plt.scatter(joined['median_household_income_2017'], joined['deaths-corrected'],)
print(joined[0:10])
plt.scatter(joined['median_household_income_2017'], joined['deaths-corrected'])
# plt.show()


X = np.array(joined['median_household_income_2017']).reshape(-1, 1)
y = np.array(joined['deaths-corrected']).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

#reg = LinearRegression().fit(X, y)

#print(reg.score(X_test, y_test))

# plt.plot(X, reg.predict(X)) #- .29

mymodel = np.poly1d(np.polyfit(np.array(X).flatten(),
                               np.array(y).flatten(), 5))

myline = np.linspace(25000, 120000)

plt.plot(myline, mymodel(myline), color="red")

plt.show()
