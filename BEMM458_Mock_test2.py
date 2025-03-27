#######################################################################################################################################################
# 
# Name:Chenyang Xu
# SID:740099095
# Exam Date:2025.3.27
# Module:BEMM458 Programming for Business Analytics
# Github link for this assignment:  https://github.com/UniversityExeterBusinessSchool/mock-test-2-Judy155
#
#######################################################################################################################################################
# Instruction 1. Read each question carefully and complete the scripts as instructed.

# Instruction 2. Only ethical and minimal use of AI is allowed. You may use AI to get advice on tool usage or language syntax, 
#                but not to generate code. Clearly indicate how and where you used AI.

# Instruction 3. Include comments explaining the logic of your code and the output as a comment below the code.

# Instruction 4. Commit to Git and upload to ELE once you finish.

#######################################################################################################################################################
import numpy as np
import pandas as pd
# Question 1 - Loops and Lists
# You are given a list of numbers representing weekly sales in units.给你一个数字列表，表示每周的销售量。
weekly_sales = [120, 85, 100, 90, 110, 95, 130]

# Write a for loop that iterates through the list and prints whether each week's sales were above or below the average sales for the period.
#编写一个for循环，遍历列表并输出每周的销售额是高于还是低于同期的平均销售额。
# Calculate and print the average sales.
#计算并打印平均销售额。
#10mins

average_sales = np.mean(weekly_sales)
print("average_sales :",average_sales)
for i in range(len(weekly_sales)):
    if weekly_sales[i]<average_sales:
        print("Week" +str(i+1)+ "sales are lower than the average sales for the same period")
    elif weekly_sales[i]>average_sales:
        print("Week" +str(i+1)+ "sales are higher than the average sales for the same period")
    else:
        print("Week" +str(i+1)+ "sales are equal to the average sales for the same period")
"""
average_sales = sum(weekly_sales) / len(weekly_sales) 题目没有import numpy！
for sales in weekly_sales:
    if sales > average_sales:
        print(f"Week's sales of {sales} is above average.")
    else:
        print(f"Week's sales of {sales} is below average.")
print(f"Average sales: {average_sales}")
"""
#######################################################################################################################################################
#7mins
# Question 2 - String Manipulation
# A customer feedback string is provided:
customer_feedback = """The product was good but could be improved. I especially appreciated the customer support and fast response times."""

# Find the first and last occurrence of the words 'good' and 'improved' in the feedback using string methods.
# Store each position in a list as a tuple (start, end) for both words and print the list.
#查找单词“good”和“improved”在反馈中第一次和最后一次出现的字符串方法。
#将两个单词在列表中的每个位置存储为元组（start, end），并打印列表。

positions = [
    (customer_feedback.find('good'), customer_feedback.find('good') + len('good')),
    (customer_feedback.find('improved'), customer_feedback.find('improved') + len('improved'))
]
print("Positions of keywords:", positions)

""""
错的
good_first= customer_feedback.find('good')
print(good_first)
good_last=customer_feedback.rfind('good')
print(good_last)

improved_first= customer_feedback.find('improved')
print(improved_first)
improved_last=customer_feedback.rfind('improved')
print(improved_last)

tuple1=(good_first,good_last)
tuple2=(improved_first,improved_last)
list1=[tuple1,tuple2]
print(list1)

"""
#######################################################################################################################################################

# Question 3 - Functions for Business Metrics
# Define functions to calculate the following metrics, and call each function with sample values (use your student ID digits for customization).
#定义函数来计算以下指标，并使用样例值调用每个函数（使用您的学生ID数字进行定制）

# 1. Net Profit Margin: Calculate as (Net Profit / Revenue) * 100.
# 2. Customer Acquisition Cost (CAC): Calculate as (Total Marketing Cost / New Customers Acquired).
# 3. Net Promoter Score (NPS): Calculate as (Promoters - Detractors) / Total Respondents * 100.
# 4. Return on Investment (ROI): Calculate as (Net Gain from Investment / Investment Cost) * 100.
# 1。净利润率：计算为（净利润/收入）* 100。
# 2。客户获取成本（CAC）：计算为（总营销成本/获得的新客户）。
# 3。净推荐分数（NPS）：计算为（推荐者-诋毁者）/总受访者* 100。
# 4。投资回报率（ROI）：计算为（投资净收益/投资成本）* 100。


def net_profit_margin(net_profit, revenue):
    return (net_profit / revenue) * 100

def customer_acquisition_cost(total_marketing_cost, new_customers_acquired):
    return total_marketing_cost / new_customers_acquired

def net_promoter_score(promoters, detractors, total_respondents):
    return ((promoters - detractors) / total_respondents) * 100

def return_on_investment(net_gain, investment_cost):
    return (net_gain / investment_cost) * 100

#740099095
# Call functions with sample values (replace with ID digits)使用样例值调用函数（用ID数字代替）
print("Net Profit Margin:", net_profit_margin(7400, 99095))
print("Customer Acquisition Cost:", customer_acquisition_cost(7400, 90))
print("Net Promoter Score:", net_promoter_score(90, 40, 990))
print("Return on Investment:", return_on_investment(4009, 99095))

#######################################################################################################################################################
#7mins
# Question 4 - Data Analysis with Pandas
# Using a dictionary sales_data, create a DataFrame from this dictionary, and display the DataFrame.
# Write code to calculate and print the cumulative monthly sales up to each month.
#使用字典sales_data，从字典中创建一个DataFrame，并显示该DataFrame。
#编写代码计算并打印每月累计销售额。

import pandas as pd

sales_data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'], 'Sales': [200, 220, 210, 240, 250]}


df = pd.DataFrame(sales_data)
df['Cumulative Sales'] = df['Sales'].cumsum()#.cumsum() 是 Pandas 的 累积求和方法
print(df)

#######################################################################################################################################################
# Question 5 - Linear Regression for Forecasting线性回归预测
# Using the dataset below, create a linear regression model to predict the demand for given prices.
# Predict the demand if the company sets the price at £26. Show a scatter plot of the data points and plot the regression line.
#使用下面的数据集，创建一个线性回归模型来预测给定价格的需求。
#如果公司把价格定在26英镑，预测一下需求。显示数据点的散点图并绘制回归线。

# Price (£): 15, 18, 20, 22, 25, 27, 30
# Demand (Units): 200, 180, 170, 160, 150, 140, 130

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data for linear regression
prices = np.array([15, 18, 20, 22, 25, 27, 30]).reshape(-1, 1) #是价格列表，需要转换为二维数组（reshape(-1,1)）以适应 sklearn 模型输入格式
demand = np.array([200, 180, 170, 160, 150, 140, 130])

# Create and train model
model = LinearRegression()
model.fit(prices, demand)#用 prices 作为自变量，demand 作为因变量，训练模型

# Predict demand at price 26
predicted_demand = model.predict(np.array([[26]]))#sklearn 线性回归模型的 predict() 方法要求输入为二维数组
print(f"Predicted demand at price £26: {predicted_demand[0]}")

# Plotting the data points and regression line
plt.scatter(prices, demand, color='blue')
plt.plot(prices, model.predict(prices), color='red')
plt.xlabel("Price (£)")
plt.ylabel("Demand (Units)")
plt.title("Price vs Demand")
plt.show()


#######################################################################################################################################################
#15mins

# Question 6 - Error Handling
# You are given a dictionary of prices for different products.给你一本不同产品价格的字典。
prices = {'A': 50, 'B': 75, 'C': 'unknown', 'D': 30}

# Write a function to calculate the total price of all items, handling any non-numeric values by skipping them.
# Include error handling in your function and explain where and why it’s needed.
#编写一个函数来计算所有商品的总价格，跳过所有非数字值。
#在你的函数中包含错误处理，并解释在哪里以及为什么需要它。

#try-except 处理错误，确保程序不会因无效数据崩溃。

def calculate_total_price(prices_dict):
    total_price = 0
    for item, price in prices_dict.items(): #items() 方法返回 (key, value) 对，即：item 代表商品名称（如 'A'、'B'），price 代表商品的价格（可能是数值，也可能是非数值，如 'unknown'）。
        try:  #尝试将 price 转换为浮点数 float(price)：如果 price 是数字（如 50、75），则转换成功，并将其加到 total_price 中。如果 price 不是数值（如 'unknown'），则 float(price) 会抛出 ValueError，跳转到 except 处理。
            total_price += float(price)
        except ValueError:
            print(f"Skipping non-numeric price for item {item}")
    return total_price

print("Total Price:", calculate_total_price(prices))
""""
为什么需要错误处理？

在 prices_dict 字典中，价格的值可能是：
	•	数值（如 50、75）
	•	非数值字符串（如 'unknown'）
	•	其他不兼容的数据类型（如 None、列表、字典）

如果我们不使用错误处理，代码会在遇到 无法转换为 float 的值 时崩溃。例如：float('unknown')  # ValueError: could not convert string to float: 'unknown'
这会导致程序 异常终止，无法继续执行。
为什么需要错误处理？
	1.	防止程序崩溃
	•	没有 try-except 时，程序在遇到 'unknown' 这样的无效数据时会 报错并停止执行。
	•	try-except 确保 即使出现错误，也不会影响整个计算过程，程序仍然可以正常运行。
	2.	跳过无效数据，不影响正确数据的计算
	•	如果一个商品价格无效，我们 只跳过该商品，而不是影响所有商品的计算。
	•	这样可以 保证总价格的计算仍然尽可能准确。
	3.	提高程序的可读性和用户体验
	•	print(f"Skipping non-numeric price for item {item}") 提示用户 哪些数据有问题，方便后续检查和修正数据。
	•	让代码更 健壮（robust），可以处理各种输入情况，而不会因为异常数据而崩溃。

def func_total_price(prices):
    total=0
    for value in prices.values():
        if type(value)==int:
            total+=value
        else:
            continue
        return total
"""




#######################################################################################################################################################

# Question 7 - Plotting and Visualization
# Generate 50 random numbers between 1 and 500, then:
# Plot a histogram to visualize the distribution of these numbers.
# Add appropriate labels for the x-axis and y-axis, and include a title for the histogram.
#生成1到500之间的50个随机数，然后：
#绘制一个直方图来可视化这些数字的分布。
#为x轴和y轴添加适当的标签，并为直方图添加标题。

import matplotlib.pyplot as plt
import random

# 生成 50 个随机数
random_numbers = [random.randint(1, 500) for _ in range(50)]

print(random_numbers)

plt.hist(random_numbers, bins=10, color='blue', edgecolor='black')
#bins：直方图的柱子（bin）个数，即数据被分成 10 个区间
#color='blue'，设置柱子的填充颜色为 蓝色。
#edgecolor='black'，设置柱子边缘颜色为 黑色，增强视觉对比。
plt.title('Histogram of 50 Random Numbers between 1 and 500')
plt.xlabel('value')
plt.ylabel('frequency')
plt.show()


#######################################################################################################################################################

# Question 8 - List Comprehensions
# Given a list of integers representing order quantities.给定一个表示订单数量的整数列表。
quantities = [5, 12, 9, 15, 7, 10]

# Use a list comprehension to create a new list that doubles each quantity that is 10 or more.
# Print the original and the new lists.
#使用列表推导式创建一个新的列表，将每个数量加倍为10或更多。
#打印原始列表和新列表。
new_list=[i*2 for i in quantities if i >= 10]

print("Original quantities:",quantities)
print("Doubled quantities:",new_list)


#######################################################################################################################################################

# Question 9 - Dictionary Manipulation
# Using the dictionary below, filter out the products with a rating of less than 4 and create a new dictionary with the remaining products.
#使用下面的字典，过滤掉评级低于4的产品，并使用剩余的产品创建一个新字典。

ratings = {'product_A': 4, 'product_B': 5, 'product_C': 3, 'product_D': 2, 'product_E': 5}

filtered_dict = {k: v for k, v in ratings.items() if v != 4} #k, v 分别表示字典的 键（产品名称） 和 值（评分）。 字典推导式
print(filtered_dict)

#filtered_ratings = {product: rating for product, rating in ratings.items() if rating >= 4}
#print("Filtered ratings:", filtered_ratings)

#######################################################################################################################################################

# Question 10 - Debugging and Correcting Code调试和纠正代码
# The following code intends to calculate the average of a list of numbers, but it contains errors:
#下面的代码打算计算一个数字列表的平均值，但是它包含错误：

values = [10, 20, 30, 40, 50]
total = 0
for i in values:
    total = total + i
average = total / len(values)
#print("The average is" + average)
print("The average is",average) #不能用加号，字符串和float不能连。应该用逗号。。逗号的英文是 comma。

# Identify and correct the errors in the code.识别并纠正代码中的错误。
# Comment on each error and explain your fixes.注释每个错误并解释你的修复。


#######################################################################################################################################################
