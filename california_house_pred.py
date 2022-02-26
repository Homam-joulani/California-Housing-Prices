import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import numpy.random as rnd
rnd.seed(0)

def load_housing_data():
    return pd.read_csv("housing.csv")

housing = load_housing_data()

housing.head(10)

housing.info()

housing["ocean_proximity"].value_counts()

housing.describe()

housing.hist( bins=50, figsize=(20,15) )

housing['income_cat'] = pd.cut(housing['median_income']
                              , bins=[0.,1.5,3.0,4.5,6.0,np.inf]
                              , labels=[1,2,3,4,5])
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit


splitter = StratifiedShuffleSplit(n_splits=1, test_size =0.2, random_state=0 )

for train_index, test_index in splitter.split(housing, housing["income_cat"] ):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts()/len(strat_test_set)

for s in (strat_train_set , strat_test_set):
    s.drop("income_cat" , axis=1 , inplace=True)
strat_train_set

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

corr_matrix= housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing=strat_train_set.drop("median_house_value",axis=1)
housing_labels=  strat_train_set["median_house_value"].copy()
housing.info()

housing2=strat_test_set.drop("median_house_value",axis=1)
housing_labels2=strat_test_set["median_house_value"].copy()

housing_labels

from sklearn.base import BaseEstimator, TransformerMixin


rooms_ix ,bedrooms_ix, population_ix, households_ix = 3,4,5,6,

class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    
    
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
        
    def fit(self,X,y=None):
        return self
    
    def transform (self,X,y=None):
        rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
        population_per_household = X[:,population_ix]/X[:,households_ix]
        
        if self.add_bedrooms_per_room :
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else: 
            return np.c_[X,rooms_per_household,population_per_household]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.impute import SimpleImputer

housing_num = housing.drop("ocean_proximity" , axis=1)

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                        ('attribs_adder', CombinedAttributeAdder()),
                        ('std_adder',StandardScaler())])
print(housing_num)
print("++++++++++++++++++++++++++++++++++++++++++++++++")
housing_num_tr= num_pipeline.fit_transform(housing_num)

print(housing_num_tr.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_attribs= list(housing_num)

# print(num_attribs)
cat_attribs = ["ocean_proximity"]

full_pipeline= ColumnTransformer([("num", num_pipeline ,num_attribs ),
                                ('cat',OneHotEncoder(),cat_attribs)])
housing

housing_prepared = full_pipeline.fit_transform(housing)

from sklearn.linear_model import LinearRegression

lin=LinearRegression()

lin.fit(housing_prepared,housing_labels)

housing2_prepared=full_pipeline.fit_transform(housing2)

pred=lin.predict(housing2_prepared)

from sklearn.metrics import mean_squared_error
lin_mse  = mean_squared_error(housing_labels2,pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

from  sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid =[{'n_estimators':[3,10,30],'max_features':[2,4,6,8,10]}
            ,{'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4,6] }]


forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg ,param_grid , cv = 5 , scoring="neg_mean_squared_error", return_train_score=True )

grid_search.fit(housing_prepared, housing_labels)

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value" , axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions =final_model.predict(X_test_prepared)

final_mse= mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
