import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import dill
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

session = pd.read_csv('data\ga_sessions.csv')
session = session.drop_duplicates()
hits = pd.read_csv('data\ga_hits.csv')
hits = hits.drop_duplicates()
df = pd.DataFrame()
x_test = pd.DataFrame()
x_train = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()

def main():

    def filter_data_session():
        global session
        columns_to_drop = [
            'device_model',
            'client_id'
        ]
        session = session.drop(columns_to_drop, axis=1)
        return session

    def filter_data_session2():
        global session
        columns_to_drop = [
            'visit_date',
            'visit_time'
        ]
        session = session.drop(columns_to_drop, axis=1)
        return session

    def date_time():
        global session
        session['month'] = session.visit_date.apply(lambda x: int(x.split('-')[1]))
        session['day'] = session.visit_date.apply(lambda x: int(x.split('-')[2]))
        session['hour'] = session.visit_time.apply(lambda x: int(x.split(':')[0]))
        return session

    def smothing():
        global session
        def calculate_outliers_3sigma(df):
            #stat = df.describe().values
            (left, write) = (round(df.mean() - df.std() * 3), round(df.mean() + df.std() * 3))
            return(left, write)

        session = session[session['visit_number'] < calculate_outliers_3sigma(session.visit_number)[1]]
        return session

    def nunique():
        level = [0.99, 0.98, 0.95, 0.9]
        for j in categorical_features_sessios:
            if j not in ['session_id', 'visit_date', 'visit_time', 'device_model', 'client_id'] and session[j].nunique() > 100:
                column = session[j].value_counts(normalize=True).to_frame().reset_index()
                sum = 0
                for q in range(4):
                    i = 0
                    sum = 0
                    for i in range(1000):
                        sum = sum + column.iloc[i, 1]
                        if sum >= level[q]:
                            break
                    if i < 100:
                        break
                m = []
                #print('j = ', j, '; i = ', i,'; level[q] = ', level[q])
                i1 = 0
                for i1 in range(i):
                    m.append(column.iloc[i1, 0])
                session[j] = session[j].apply(lambda x: x if x in m else "other")

    def filter_data_hits():
        global hits
        columns_to_drop = [
            'hit_date', 'hit_time', 'hit_number', 'hit_type',
            'hit_referer', 'hit_page_path', 'event_category', 'event_label', 'event_value'
        ]
        hits = hits.drop(columns_to_drop, axis=1)
        return session

    def dropna():
        global hits
        hits = hits.dropna()
        return hits

    def hit_page_path():
        global hits
        hits['page'] = hits.hit_page_path.apply(lambda x: x.split('/')[0])

    def nunique_hits():
        global hits
        level = [0.99, 0.98, 0.95, 0.9]
        if hits['event_label'].nunique() > 100:
            column = hits['event_label'].value_counts(normalize=True).to_frame().reset_index()
            sum = 0
            for q in range(4):
                i = 0
                sum = 0
                for i in range(1000):
                    sum = sum + column.iloc[i, 1]
                    if sum >= level[q]:
                        break
                if i < 100:
                    break
            m = []
            #print('j = ', j, '; i = ', i,'; level[q] = ', level[q])
            for i1 in range(i):
                m.append(column.iloc[i1, 0])
            hits = hits.query("event_label in %s" % m)

    def action():
        global hits
        m = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
             'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
             'sub_submit_success', 'sub_car_request_submit_click']
        hits['action'] = hits.event_action.apply(lambda x: 1 if x in m else 0)

    def filter_data_hits2():
        global hits
        hits = hits.drop(['hit_date', 'hit_type', 'hit_page_path', 'event_action'], axis=1)
        return hits

    def device_os():
        global session
        session['device_os'] = session['device_os'].fillna('Android')

    def merge_and_filter():
        global hits, session, df
        df = session.merge(hits, on='session_id')
        df.drop_duplicates()
        if df.shape[0] > 1000000:
            df = df[(df.session_id > '9') | (df.action == 1)]
        df = df.drop(['session_id', 'event_action'], axis=1)
        return df

    def missing_values(df):
        missing_values = round(((df.isna().sum() / len(df)) * 100).sort_values(), 2)
        print('Процент пропущенных значений ')
        print(missing_values)

    def num_unique(df):
        n = list(df.columns)
        for i in n:
            print(i, '\t', df[i].nunique())

    def xy():
        global x_train, x_test, y_train, y_test
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


    numerical_features_session = session.select_dtypes(include=['int64', 'float64']).columns
    categorical_features_sessios = session.select_dtypes(include=['object']).columns
    #numerical_features_hits = hits.select_dtypes(include=['int64', 'float64']).columns
    #categorical_features_hits = hits.select_dtypes(include=['object']).columns

    numerical_transformer_session = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer_session = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor_session = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer_session, numerical_features_session),
        ('categorical', categorical_transformer_session, categorical_features_sessios)
    ])

    clean_data = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data_session())),         # удалить device_model, client_id
        ('preprocessor', preprocessor_session),                         # заполнить пустые ячейки
        ('device_os', device_os()),
        ('nunique', FunctionTransformer(nunique())),                    # оставляем популярные значения
        ('date_time', FunctionTransformer(date_time())),                # обрабатываем дату и время
        ('smotuing', FunctionTransformer(smothing())),                  # visit_number убрать выбросы
        ('filter2', FunctionTransformer(filter_data_session2())),       # удалить visit_date, visit_time
        ('filter_data_hits', FunctionTransformer(filter_data_hits())),  # оставить только session_id, event_action
        #('dropna', FunctionTransformer(dropna())),                      #
        #('hit_page_path', FunctionTransformer(hit_page_path())),
        #('nunique_hits', FunctionTransformer(nunique_hits())),
        ('action', FunctionTransformer(action())),                      # создать целевое поле action
        #('filter_data_hits2', FunctionTransformer(filter_data_hits2())),
        ('merge', FunctionTransformer(merge_and_filter()))              # создать df, удалить session_id, event_action
    ])

    X = df.drop('action', axis=1)
    Y = df['action']
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features),
    ])

    models = (
        LogisticRegression(max_iter=5000, random_state=42),
        RandomForestClassifier(),
        #MLPClassifier()
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('XY', xy()),
            ('classifier', model)
        ])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        acc = accuracy_score(y_test, pred)
        matr = confusion_matrix(y_test, pred)
        print(f'model: {type(model).__name__}, acc: {acc:.4f}')
        print(f'confusion_matrix: {matr}')
        #score = cross_val_score(pipe, X, Y, cv=4, scoring='accuracy')
        #print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        roc = roc_auc_score(Y, pipe.predict_proba(X)[:, 1])
        print(f'ROC-AUC: {roc}\n')


        if acc > best_score:
            best_score = acc
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    best_pipe.fit(X, Y)
    file_name = 'sberavto2.pkl'
    with open(file_name, 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                "name": "Target action prediction model",
                "author": "Oleg Dik",
                "version": 1,
                "date": datetime.now(),
                "type": type(best_pipe.named_steps["classifier"]).__name__,
                "accuracy": best_score
            }
        }, file)


if __name__ == '__main__':
    main()