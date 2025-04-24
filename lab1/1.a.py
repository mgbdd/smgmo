import numpy as np
import matplotlib.pyplot as plt
import progression as pr

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 

X = 1
COEF = 3
N = 10
EPSILON = 0.5

def generate_x():
    return np.random.uniform(-X, X)

def generate_eps():
    return np.random.uniform(-EPSILON, EPSILON)

def generate_coef():
    a = np.random.uniform(-COEF, COEF)
    b = np.random.uniform(-COEF, COEF)
    c = np.random.uniform(-COEF, COEF)
    d = np.random.uniform(-COEF, COEF)
    return a, b, c, d

def f(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def draw_function(a, b, c, d):
    x = np.linspace(-5, 5, 200)
    y = f(x, a, b, c, d)
    fig = plt.figure(figsize=(8, 6)) 
    plt.plot(x, y, label=f'{np.round(a, 3)}x^3 + {np.round(b, 3)}x^2 + {np.round(c, 3)}x + {np.round(d, 3)}')  
    plt.title('График функции и выборки') 
    plt.xlabel('x') 
    plt.ylabel('f(x)')  
    plt.text(-2.5, -3.5, f'N={N}, eps0={EPSILON}', fontsize=8, color='blue') 
    plt.grid(True)  
    plt.legend() 
    plt.xlim(-3, 3) 
    plt.ylim(-5, 5) 
    return fig

def main():
    print("(x_i, y_i):")
    a, b, c, d = generate_coef()
    #fig = draw_function(a, b, c, d)
    x_list = list()
    y_list = list()
    for i in range(N):
        x_i = generate_x()
        eps = generate_eps()
        y_i = f(x_i, a, b, c, d) + eps
        x_list.append(x_i)
        y_list.append(y_i)
        print("i:", i, " (x_i, y_i):", np.round(x_i, 3), np.round(y_i, 3))
        #plt.scatter(x_i, y_i, color='red', label='Выборка' if i == 0 else "")
    
    #plt.legend()
    #plt.show()
    
    x = np.array(x_list)  
    y = np.array(y_list)  

    # разделение данных на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # подготовка данных 
    x_train = x_train.reshape(-1, 1)  # преобразуем в формат (n_samples, n_features) - столбец
    x_test = x_test.reshape(-1, 1)

    # перебор степеней полинома
    degrees = [6]
    models = []  # для хранения обученных моделей
    mse_train = []  # MSE на обучающей выборке
    mse_test = []  # MSE на тестовой выборке

    for degree in degrees:
        # создание и обучение модели
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x_train, y_train)
        models.append(model)

        # оценка качества (MSE)
        y_train_pred = model.predict(x_train)
        mse_train.append(mean_squared_error(y_train, y_train_pred))

        y_test_pred = model.predict(x_test)
        mse_test.append(mean_squared_error(y_test, y_test_pred))

    # визуализация результатов
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, color='red', label='Выборка')  # отображаем исходные данные

    x_true = np.linspace(-5, 5, 200) # отображаем оригинальную функцию
    plt.plot(x_true, f(x_true, a, b, c, d), color='black', linestyle='--', label='Истинная функция', alpha=0.7)
    x_plot = np.linspace(-5, 5, 200)  
    x_plot = x_plot.reshape(-1, 1)

    for i, degree in enumerate(degrees):
        y_plot = models[i].predict(x_plot)
        plt.plot(x_plot, y_plot, label=f'Полином степени {degree}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Восстановление функциональной зависимости')
    plt.grid(True)
    plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5) 
    plt.show()

    # дополнительный анализ: Выбор "оптимальной" степени на тестовых данных:
    best_degree_index = np.argmin(mse_test)
    best_degree = degrees[best_degree_index]
    best_model = models[best_degree_index]

    print(f"Оптимальная степень полинома (на основе MSE на тестовой выборке): {best_degree}")
    print(f"MSE на тестовой выборке для оптимальной степени: {mse_test[best_degree_index]:.2f}")

    # вывод коэффициентов лучшей модели
    if best_degree > 0:
      print(f"\nКоэффициенты лучшей модели (степень {best_degree}):")
      print(best_model.named_steps['linearregression'].coef_) 
      print(f"Свободный член: {best_model.named_steps['linearregression'].intercept_}") 
    return

if __name__ == '__main__':
    main()